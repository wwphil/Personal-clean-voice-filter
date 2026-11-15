"""Phase 0 gate: ONNX export of BSRNN-TSE (runtime form) + ECAPA (enrollment).

Verifies:
  1. torch.onnx.export succeeds (complex STFT/ISTFT, LSTM, GroupNorm, dynamic T).
  2. Parity: ONNX Runtime (CPU, fp32) vs torch (CPU, fp32) output.
  3. CPU RTF of the ONNX path at 1/2/3 s chunks, with 50 % overlap context,
     at 12 threads (this box) and 4 threads (laptop-class proxy).

Usage:
    .venv/bin/python tools/phase0_onnx.py
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
OUT_DIR = ROOT / "models" / "onnx"


# ------------------------------------------------------------------- helpers
def speech_like_16k(n_samples: int, sr: int = 16000, seed: int = 0) -> np.ndarray:
    """Speech-like signal: amplitude-modulated harmonics + broadband noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sr
    f0 = 110 + 40 * np.sin(2 * np.pi * 0.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = np.zeros(n_samples)
    for k in range(1, 13):
        sig += (1.0 / k) * np.sin(k * phase + rng.uniform(0, 6.28))
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    am = np.clip(am, 0.05, 1.0)
    noise = rng.normal(0, 0.02, n_samples)
    x = 0.6 * sig * am + noise
    return (x / (np.abs(x).max() + 1e-9) * 0.5).astype(np.float32)


def bench(fn, iters: int = 3) -> float:
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ------------------------------------------------------------------- loaders
def load_tse():
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
    return model


class TSERuntime(torch.nn.Module):
    """BSRNN in production runtime form: fixed enrollment embedding in."""

    def __init__(self, bsrnn):
        super().__init__()
        self.bsrnn = bsrnn

    def forward(self, wav, spk_emb):
        # wav [1, T] @16k, spk_emb [1, 192]
        out, _ = self.bsrnn(wav, spk_emb)
        return out


def make_runtime_tse(model):
    """Production runtime form: fixed enrollment embedding in, speaker
    encoder bypassed (Identity). Returns (wrapper, emb, ecapa)."""
    import torchaudio

    sr = 16000
    enroll = torch.from_numpy(speech_like_16k(sr * 6, sr)).unsqueeze(0)
    fbank = torchaudio.compliance.kaldi.fbank(
        enroll, num_mel_bins=80, sample_frequency=sr, dither=0.0,
        frame_length=25, frame_shift=10,
    )
    fbank = fbank - fbank.mean(0)  # CMN
    feats = fbank.unsqueeze(0)  # [1, T, 80]
    ecapa = model.spk_model  # keep original before swapping
    with torch.no_grad():
        emb = ecapa(feats)
    emb = emb[-1] if isinstance(emb, tuple) else emb
    emb = emb.detach()  # [1, 192]
    model.spk_model = torch.nn.Identity()

    return TSERuntime(model), emb, ecapa


class EcapaWrap(torch.nn.Module):
    def __init__(self, spk_model):
        super().__init__()
        self.m = spk_model

    def forward(self, feats):
        out = self.m(feats)  # feats [1, T, 80]
        out = out[-1] if isinstance(out, tuple) else out
        return out


# ------------------------------------------------------------------- export
# torch 2.10: neither the dynamo exporter (unbacked symbols inside
# torch.istft length math) nor the legacy exporter (dynamic reshape) can
# export BSRNN with a dynamic time axis. The app uses a fixed, small set
# of window sizes per device tier, so export one static graph each.
WINDOW_SAMPLES = (24000, 48000, 72000)  # 1/2/3 s chunks + 50 % overlap


def _postfix(path: Path) -> None:
    """torch 2.10 dynamo exporter emits int32 arange initializers used as
    Gather data, which makes ScatterND indices int32 -> ORT rejects the
    graph. Convert such initializers to int64 and drop stale value_info
    annotations (ORT re-infers types)."""
    import onnx
    import onnx.numpy_helper as nh

    m = onnx.load(str(path))
    graph = m.graph
    gather_data = {n.input[0] for n in graph.node if n.op_type == "Gather"}
    for i in graph.initializer:
        if i.name in gather_data and nh.to_array(i).dtype == np.int32:
            i.CopyFrom(nh.from_array(nh.to_array(i).astype(np.int64), i.name))
    graph.value_info.clear()
    onnx.save(m, str(path))


def export_tse(rt, emb, path: Path, n_samples: int, opset: int) -> None:
    wav = torch.from_numpy(
        speech_like_16k(n_samples, seed=n_samples % 97)).unsqueeze(0)
    torch.onnx.export(
        rt,
        (wav, emb),
        str(path),
        input_names=["wav", "spk_emb"],
        output_names=["out"],
        opset_version=opset,
    )
    _postfix(path)
    print(f"exported {path.name} (opset {opset}, T={n_samples})")


def export_ecapa(spk_model, path: Path, n_frames: int, opset: int) -> None:
    wrap = EcapaWrap(spk_model)
    feats = torch.zeros(1, n_frames, 80, dtype=torch.float32)
    # Legacy (TorchScript) exporter: the dynamo exporter's constraint solver
    # trips on an int32 size guard inside ECAPA (T <= 2^31/4608).
    torch.onnx.export(
        wrap,
        (feats,),
        str(path),
        input_names=["fbank"],
        output_names=["emb"],
        dynamic_axes={"fbank": {1: "T"}},
        opset_version=min(opset, 17),
        dynamo=False,
    )
    _postfix(path)
    print(f"exported {path.name} (opset {min(opset, 17)}, dynamic frames)")


# ------------------------------------------------------------------ parity
def check_parity(rt, emb, onnx_path: Path, n_samples: int) -> bool:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    wav = torch.from_numpy(
        speech_like_16k(n_samples, seed=n_samples % 199)).unsqueeze(0)
    with torch.no_grad():
        ref = rt(wav, emb).numpy()
    got = sess.run(["out"], {"wav": wav.numpy(), "spk_emb": emb.numpy()})[0]
    diff = np.abs(ref - got).max()
    cos = cosine(ref, got)
    ok = bool(diff < 1e-3 and cos > 0.9999)
    print(f"  TSE parity  {n_samples:>6} smp: max|diff| {diff:.3e}  "
          f"cosine {cos:.9f}  {'OK' if ok else 'FAIL'}")
    return ok


def check_ecapa_parity(spk_model, onnx_path: Path) -> bool:
    import onnxruntime as ort
    import torchaudio

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    wrap = EcapaWrap(spk_model)
    wrap.eval()
    ok = True
    for secs in (3.0, 6.0, 15.0):
        n = int(secs * 16000)
        wav = torch.from_numpy(speech_like_16k(n, seed=int(secs * 7))).unsqueeze(0)
        fb = torchaudio.compliance.kaldi.fbank(
            wav, num_mel_bins=80, sample_frequency=16000, dither=0.0,
            frame_length=25, frame_shift=10,
        )
        fb = fb - fb.mean(0)
        feats = fb.unsqueeze(0)
        with torch.no_grad():
            ref = wrap(feats).numpy()
        got = sess.run(["emb"], {"fbank": feats.numpy()})[0]
        diff = np.abs(ref - got).max()
        cos = cosine(ref, got)
        status = "OK" if diff < 1e-3 and cos > 0.9999 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  ECAPA parity {secs:>4.1f}s: max|diff| {diff:.3e}  "
              f"cosine {cos:.9f}  {status}")
    return ok


# --------------------------------------------------------------------- RTF
def onnx_rtf(onnx_path: Path, emb, threads: int, n_samples: int,
             chunk_secs: float, overlap: float = 0.5) -> None:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so,
                                providers=["CPUExecutionProvider"])
    sr = 16000
    wav = speech_like_16k(n_samples, sr, seed=n_samples % 89).reshape(1, -1)
    emb_np = emb.numpy()
    n = int(chunk_secs * sr)

    def run():
        sess.run(["out"], {"wav": wav, "spk_emb": emb_np})

    run()  # warmup
    dt = bench(run, iters=3)
    rtf = (n / sr) / dt
    latency = (n + 2 * int(overlap * n)) / sr
    print(f"  {threads:>2} thr  {chunk_secs:>4.1f}s chunk: {dt*1000:>8.1f} ms"
          f"  RTF {rtf:>6.2f}  latency {latency:>5.2f}s")


# -------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rtf", action="store_true")
    ap.add_argument("--skip-export", action="store_true",
                    help="reuse existing .onnx files (no export/parity)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_tse()
    rt, emb, ecapa = make_runtime_tse(model)
    print(f"runtime emb shape: {tuple(emb.shape)}")

    if args.skip_export:
        ok_tse = True
        for ws in WINDOW_SAMPLES:
            if not (OUT_DIR / f"tse_bsrnn_{ws}.onnx").exists():
                ok_tse = False
        ok_ecapa = (OUT_DIR / "ecapa_c512.onnx").exists()
    else:
        ok_tse = True
        for ws in WINDOW_SAMPLES:
            path = OUT_DIR / f"tse_bsrnn_{ws}.onnx"
            try:
                export_tse(rt, emb, path, ws, 20)
            except Exception as e:  # noqa: BLE001
                print(f"TSE export {ws} failed: {type(e).__name__}: "
                      f"{str(e).splitlines()[0][:200]}")
                ok_tse = False
                continue
            if not check_parity(rt, emb, path, ws):
                ok_tse = False

        ecapa_path = OUT_DIR / "ecapa_c512.onnx"
        try:
            export_ecapa(ecapa, ecapa_path, 600, 20)
            ok_ecapa = check_ecapa_parity(ecapa, ecapa_path)
        except Exception as e:  # noqa: BLE001
            print(f"ECAPA export failed: {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:200]}")
            ok_ecapa = False

    if not args.skip_rtf:
        for ws, cs in zip(WINDOW_SAMPLES, (1.0, 2.0, 3.0)):
            path = OUT_DIR / f"tse_bsrnn_{ws}.onnx"
            if not path.exists():
                continue
            print(f"\n== TSE ONNX RTF, window {ws} smp "
                  f"({cs:.0f}s chunk + 50% overlap) ==")
            onnx_rtf(path, emb, 12, ws, cs)
            onnx_rtf(path, emb, 4, ws, cs)

    print(f"\nGATE: parity TSE={'PASS' if ok_tse else 'FAIL'}  "
          f"ECAPA={'PASS' if ok_ecapa else 'FAIL'}")
    sys.exit(0 if (ok_tse and ok_ecapa) else 2)


if __name__ == "__main__":
    main()
