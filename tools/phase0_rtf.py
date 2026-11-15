"""Phase 0 spike: measure RTF of the TSE model (BSRNN), DeepFilterNet2, and Silero VAD.

Usage:
    .venv/bin/python tools/phase0_rtf.py [--tse-only] [--df-only] [--vad-only]

RTF = audio_seconds / compute_seconds. RTF <= 0.5 means we keep 50 % of the
device free while staying in real time.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

TSE_MODEL_DIR = MODEL_DIR / "wesep" / "english"


def speech_like_16k(n_samples: int, sr: int = 16000, seed: int = 0) -> np.ndarray:
    """Speech-like signal: amplitude-modulated harmonics + broadband noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sr
    # pseudo-fundamental with slow variation (male ~110 Hz, female ~220 Hz mix)
    f0 = 110 + 40 * np.sin(2 * np.pi * 0.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = np.zeros(n_samples)
    for k in range(1, 13):
        sig += (1.0 / k) * np.sin(k * phase + rng.uniform(0, 6.28))
    # syllabic amplitude modulation
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    am = np.clip(am, 0.05, 1.0)
    noise = rng.normal(0, 0.02, n_samples)
    x = 0.6 * sig * am + noise
    return (x / (np.abs(x).max() + 1e-9) * 0.5).astype(np.float32)


def bench(fn, iters: int = 3) -> float:
    """Return best-of-`iters` wall time in seconds."""
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------------------------------------------------------- TSE (BSRNN)
def load_tse(device: str):
    import yaml

    sys.path.insert(0, str(ROOT))
    from pvcf._vendor import ensure_wesep, ensure_wespeaker
    ensure_wesep()
    ensure_wespeaker()
    from wesep.models import get_model
    from wesep.utils.checkpoint import load_pretrained_model

    cfg = yaml.safe_load((TSE_MODEL_DIR / "config.yaml").read_text())
    margs = cfg["model_args"]["tse_model"]
    if "spk_model_init" in margs:
        margs["spk_model_init"] = False
    model = get_model(cfg["model"]["tse_model"])(**margs)
    load_pretrained_model(model, str(TSE_MODEL_DIR / "avg_model.pt"))
    model.eval()
    model.to(device)
    return model


def tse_rtf(device: str, chunk_secs: list[float], overlap: float = 0.0,
            cached_emb: bool = False) -> None:
    import torchaudio

    model = load_tse(device)
    sr = 16000
    total = speech_like_16k(sr * 30, sr)
    # enrollment: first 6 s, fbank features (matches training: 80-mel, 10 ms shift, CMN)
    enroll = torch.from_numpy(total[: sr * 6]).unsqueeze(0)
    fbank = torchaudio.compliance.kaldi.fbank(
        enroll, num_mel_bins=80, sample_frequency=sr, dither=0.0,
        frame_length=25, frame_shift=10,
    )
    fbank = fbank - fbank.mean(0)  # CMN
    feats = fbank.unsqueeze(0).to(device)  # [1, T, 80]

    if cached_emb:
        # cache the ECAPA embedding once (identical math, spk_transform is Identity)
        with torch.no_grad():
            emb = model.spk_model(feats)
        emb = (emb[-1] if isinstance(emb, tuple) else emb).to(device)
        model.spk_model = torch.nn.Identity()

    tag = " (cached emb)" if cached_emb else " (emb per chunk)"
    print(f"\n== TSE BSRNN on {device}{tag} ==")
    print(f"{'chunk':>6} {'ctx':>5} {'ms/chunk':>10} {'RTF':>6} {'latency(s)':>12}")
    for cs in chunk_secs:
        n = int(cs * sr)
        ctx = int(overlap * cs * sr)
        chunk = torch.from_numpy(total[: n + 2 * ctx]).unsqueeze(0).to(device)

        def run():
            with torch.no_grad():
                out = model(chunk, emb if cached_emb else feats)
            return out[0] if isinstance(out, (tuple, list)) else out

        run()  # warmup
        dt = bench(run, iters=3)
        audio_per_run = n / sr  # only the non-context part counts as new audio
        rtf = audio_per_run / dt
        latency = (n + 2 * ctx) / sr
        print(f"{cs:>5.1f}s {ctx/sr:>4.1f}s {dt*1000:>10.1f} {rtf:>6.2f} {latency:>12.2f}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


# ----------------------------------------------------------------- DF2
def df2_rtf(device: str) -> None:
    sys.path.insert(0, str(ROOT / "vendor" / "DeepFilterNet" / "DeepFilterNet"))
    import importlib

    import df
    dfe = importlib.import_module("df.enhance")  # df.enhance is shadowed by the fn

    model, df_state, _, _ = df.init_df(
        model_base_dir=None,  # pretrained, auto-downloaded from GitHub
        log_level="ERROR",
        log_file=None,
        default_model="DeepFilterNet2",
    )
    orig_get_device = dfe.get_device
    if device == "cpu":
        dfe.get_device = lambda: "cpu"  # enhance() resolves device at call time
        model = model.cpu()

    sr = df_state.sr()
    dur = 30
    # Time-domain audio stays on CPU (Rust STFT/iSTFT); only the mask model
    # runs on `device`. That is DF2's intended hybrid layout.
    x = torch.from_numpy(speech_like_16k(sr * dur, sr)).unsqueeze(0)

    print(f"\n== DeepFilterNet2 on {device} ==")
    try:
        with torch.no_grad():
            y = df.enhance(model, df_state, x)  # warmup + shape check
            assert y.shape == x.shape
            dt = bench(lambda: df.enhance(model, df_state, x), iters=3)
    finally:
        dfe.get_device = orig_get_device
    rtf = dur / dt
    print(f"{dur} s batch: {dt*1000:.1f} ms  RTF {rtf:.2f}")


# ----------------------------------------------------------------- VAD
def vad_rtf() -> None:
    from silero_vad import load_silero_vad

    sr = 16000
    audio = speech_like_16k(sr * 60, sr)
    model = load_silero_vad()  # silero 6.x: returns the callable directly
    block = 512  # 32 ms at 16 kHz

    def run():
        for i in range(0, len(audio), block):
            chunk = torch.from_numpy(audio[i : i + block])
            model(chunk, sr)  # stateful: RNN state kept in module buffers
    run()  # warmup
    dt = bench(run, iters=3)
    rtf = 60.0 / dt
    print(f"\n== Silero VAD (torch CPU) ==")
    print(f"60 s in {dt*1000:.1f} ms  RTF {rtf:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tse-only", action="store_true")
    ap.add_argument("--df-only", action="store_true")
    ap.add_argument("--vad-only", action="store_true")
    args = ap.parse_args()
    do_tse = do_df = do_vad = not (args.tse_only or args.df_only or args.vad_only)
    if args.tse_only:
        do_tse = True
    if args.df_only:
        do_df = True
    if args.vad_only:
        do_vad = True

    print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
    n = torch.get_num_threads()
    torch.set_num_threads(12)  # leave headroom on 24 cores
    print("torch cpu threads:", torch.get_num_threads())

    if do_tse:
        tse_rtf("cuda", [1.0, 2.0, 3.0], cached_emb=True)
        tse_rtf("cpu", [1.0, 2.0, 3.0], cached_emb=True)
        tse_rtf("cuda", [1.0, 2.0, 3.0], overlap=0.5, cached_emb=True)
        tse_rtf("cpu", [1.0, 2.0, 3.0], overlap=0.5, cached_emb=True)
    if do_df:
        df2_rtf("cpu")
        if torch.cuda.is_available():
            df2_rtf("cuda")
    if do_vad:
        vad_rtf()


if __name__ == "__main__":
    main()
