"""CI benchmark: ONNX Runtime CPU RTF for the TSE models.

Torch-free: only onnxruntime + numpy. Measures RTF for each static window
model at a fixed thread count, so GitHub Actions (representative x86 CPU)
can report whether the Isolated mode stays real-time.

Usage:
    python tools/ci_onnx_rtf.py --dir models/onnx [--threads 4] [--min-rtf 1.0]

Exit code 0 if every measured RTF >= --min-rtf, 1 otherwise.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import time
from pathlib import Path

import numpy as np

SR = 16000
# window samples -> (new chunk seconds at 50 % overlap)
WINDOW_CHUNK = {24000: 1.0, 48000: 2.0, 72000: 3.0}


def speech_like_16k(n_samples: int, sr: int = SR, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sr
    f0 = 110 + 40 * np.sin(2 * np.pi * 0.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = np.zeros(n_samples)
    for k in range(1, 13):
        sig += (1.0 / k) * np.sin(k * phase + rng.uniform(0, 6.28))
    am = np.clip(0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t), 0.05, 1.0)
    x = 0.6 * sig * am + rng.normal(0, 0.02, n_samples)
    return (x / (np.abs(x).max() + 1e-9) * 0.5).astype(np.float32)


def bench(fn, iters: int = 3) -> float:
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="models/onnx")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--min-rtf", type=float, default=1.0)
    args = ap.parse_args()

    d = Path(args.dir)
    files = sorted(
        p for p in d.glob("tse_bsrnn_*.onnx")
        if re.fullmatch(r"tse_bsrnn_\d+\.onnx", p.name)
    )
    if not files:
        print(f"no tse_bsrnn_*.onnx models in {d}")
        raise SystemExit(1)

    import onnxruntime as ort

    n_cpu = os.cpu_count()
    print(f"ci onnx rtf: threads={args.threads} (host cpus={n_cpu})")
    print(f"{'window':>8} {'chunk':>6} {'ms/chunk':>10} {'RTF':>7}")
    all_ok = True
    for path in files:
        ws = int(re.fullmatch(r"tse_bsrnn_(\d+)\.onnx", path.name).group(1))
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(path), so,
                                    providers=["CPUExecutionProvider"])
        wav = speech_like_16k(ws, seed=ws % 89).reshape(1, -1)
        emb = np.zeros((1, 192), dtype=np.float32)
        emb[0, 0] = 1.0  # arbitrary fixed embedding

        def run():
            sess.run(["out"], {"wav": wav, "spk_emb": emb})

        run()  # warmup
        dt = bench(run, iters=3)
        chunk = WINDOW_CHUNK.get(ws, ws / SR / 1.5)
        rtf = (chunk * SR) / SR / dt
        ok = rtf >= args.min_rtf
        all_ok = all_ok and ok
        print(f"{ws:>8} {chunk:>5.1f}s {dt*1000:>10.1f} {rtf:>7.2f} "
              f"{'OK' if ok else 'FAIL'}")

    # --- DF2 mask model (static 32-frame graph; 1 call = 30 new frames = 100 ms)
    df2 = d / "df2_mask.onnx"
    if df2.exists():
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(df2), so, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        spec = rng.normal(0, 0.1, (1, 1, 32, 481, 2)).astype(np.float32)
        erb = rng.normal(0, 0.1, (1, 1, 32, 32)).astype(np.float32)
        fspec = rng.normal(0, 0.1, (1, 1, 32, 96, 2)).astype(np.float32)

        def run():
            sess.run(["enh"], {"spec": spec, "feat_erb": erb, "feat_spec": fspec})

        run()  # warmup
        dt = bench(run, iters=3)
        rtf = 0.1 / dt  # 100 ms of new audio per call
        ok = rtf >= args.min_rtf
        all_ok = all_ok and ok
        print(f"{'df2':>8} {'0.1':>5}s {dt*1000:>10.1f} {rtf:>7.2f} "
              f"{'OK' if ok else 'FAIL'}")

    print(f"GATE: {'PASS' if all_ok else 'FAIL'} (min RTF {args.min_rtf})")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
