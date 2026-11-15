"""Export the DeepFilterNet2 mask model to ONNX (torch-free inference).

The shipped pipeline is: Rust STFT/ERB (libdf) -> ONNX mask model -> Rust iSTFT.
DF2's default checkpoint runs the deep-filter stage through `df.multiframe.DF`
(complex ops, unexportable). We swap in the parameter-free `DfOp` real-arithmetic
implementation (the paper's reference method, `real_unfold`; all DfOp methods are
mathematically equivalent per df/modules.py).

Chunk contract: the runtime processes fixed 100 ms chunks (30 frames at
160-sample hop). The graph is exported with a *static* frame count N=32
(30 new frames + 2 lookahead frames); the last 2 output frames are the
lookahead tail and are dropped by the host. The model is stateless per call
(the encoder GRU hidden state is discarded each forward), so calls are
independent — the host only keeps a rolling 2-frame overlap of the STFT
features.

Parity checks (all at T=32, fresh model state):
  1. real-ops model vs original complex-ops model (spec domain)
  2. ONNX vs torch (all outputs)
  3. end-to-end time domain through the Rust iSTFT

Usage:
    .venv/bin/python tools/export_df2.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "DeepFilterNet" / "DeepFilterNet"))

import df  # noqa: E402
from df.deepfilternet2 import DfOutputReshapeOld  # noqa: E402
from df.enhance import as_complex, df_features  # noqa: E402

OUT_PATH = ROOT / "models" / "onnx" / "df2_mask.onnx"


N_FRAMES = 32  # static frame count per call: 30 (100 ms) + 2 lookahead


class DfOpRealUnfoldStatic(torch.nn.Module):
    """DfOp.forward_real_unfold, exportable: cat-based static zero padding
    (no F.pad / no ConstantOfShape) and a static-shape view re-anchor."""

    def __init__(self, df_bins: int, df_order: int, df_lookahead: int, freq_bins: int,
                 frames: int):
        super().__init__()
        self.df_bins = df_bins
        self.df_order = df_order
        self.df_lookahead = df_lookahead
        self.freq_bins = freq_bins
        self.frames = frames
        front = df_order - df_lookahead - 1
        back = df_lookahead
        self.pads = (0, 0, 0, 0, front, back, 0, 0)  # 4D input, pad dim 1 front/back

    def forward(self, spec, coefs, alpha=None):
        x = spec[..., : self.df_bins, :].squeeze(1)  # [B, T, F, 2]
        # Re-anchor the static shape: the exporter's shape checker loses the
        # dims after the mask path and rejects the zero-padding Concat.
        x = x.view(1, self.frames, self.df_bins, 2)
        if self.pads[4] or self.pads[5]:
            # cat-based padding with constant zeros (B=1, df_bins are static):
            # new_zeros() emits ConstantOfShape, which the exporter's shape
            # checker rejects; F.pad emits a dynamic (tensor) ONNX Pad.
            front = torch.zeros(1, self.pads[4], self.df_bins, 2,
                                device=x.device, dtype=x.dtype)
            back = torch.zeros(1, self.pads[5], self.df_bins, 2,
                               device=x.device, dtype=x.dtype)
            padded = torch.cat([front, x, back], dim=1)
        else:
            padded = x
        padded = padded.unfold(dimension=1, size=self.df_order, step=1)  # [B, T, F, 2, O]
        padded = padded.permute(0, 1, 4, 2, 3)  # [B, T, O, F, 2]
        # complex sum-product over the DF window (functional, no in-place)
        spec_re = padded[..., 0] * coefs[..., 0] - padded[..., 1] * coefs[..., 1]
        spec_im = padded[..., 1] * coefs[..., 0] + padded[..., 0] * coefs[..., 1]
        spec_f = torch.stack((spec_re, spec_im), dim=-1).sum(dim=2)  # [B, T, F, 2]
        # assign_df
        spec_out = spec.clone()
        if alpha is not None:
            alpha = alpha.view(spec.shape[0], 1, -1, 1, 1)
            spec_out[..., : self.df_bins, :] = (
                spec_f.unsqueeze(1) * alpha + spec[..., : self.df_bins, :] * (1 - alpha))
        else:
            spec_out[..., : self.df_bins, :] = spec_f.unsqueeze(1)
        return spec_out


def load_swapped():
    model, df_state, *_ = df.init_df(
        model_base_dir=None, log_level="ERROR", log_file=None,
        default_model="DeepFilterNet2")
    model = model.cpu().eval()
    model.df_op = DfOpRealUnfoldStatic(
        model.df_bins, model.df_order, model.df_lookahead, model.freq_bins,
        frames=N_FRAMES)
    # This checkpoint trains with dfop_method="df" (complex MF.DF, no alpha
    # blending; dfalphaloss factor 0). Match that semantics exactly.
    model.use_alpha = False
    model.df_out_transform = DfOutputReshapeOld(model.df_order, model.df_bins)
    return model, df_state


def main() -> None:
    torch.manual_seed(0)
    N = N_FRAMES  # input frames per call: 30 (100 ms) + 2 lookahead
    model, df_state = load_swapped()
    print(f"DF2: {sum(p.numel() for p in model.parameters())/1e6:.2f} M params")

    sr = df_state.sr()
    rng = np.random.default_rng(0)
    a1 = torch.from_numpy(rng.normal(0, 0.1, (1, int(sr * 1.5))).astype(np.float32))
    s1, e1, f1 = df_features(a1, df_state, model.df_bins, device="cpu")
    s1, e1, f1 = s1[:, :, :N], e1[:, :, :N], f1[:, :, :N]

    # (1) real ops vs original complex-ops reference
    ref_model, *_ = df.init_df(
        model_base_dir=None, log_level="ERROR", log_file=None,
        default_model="DeepFilterNet2")
    ref_model = ref_model.cpu().eval()
    with torch.no_grad():
        ref_c = ref_model(s1.clone(), e1, f1)[0].numpy()
        ref_r = model(s1.clone(), e1, f1)[0].numpy()
    d1 = np.abs(ref_r - ref_c).max()
    print(f"real vs complex op: max|diff| {d1:.3e}")
    assert d1 < 2e-3, "swapped DfOp differs from reference"

    # export (legacy TorchScript exporter, static shapes)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (s1, e1, f1), str(OUT_PATH),
        input_names=["spec", "feat_erb", "feat_spec"],
        output_names=["enh", "mask", "lsnr", "df_alpha"],
        opset_version=17,
        dynamo=False,
    )
    print(f"exported {OUT_PATH.name} ({OUT_PATH.stat().st_size/1e6:.1f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUT_PATH), providers=["CPUExecutionProvider"])

    # (2) ONNX vs torch (model is stateless per call; fresh state on both sides)
    model.eval()
    with torch.no_grad():
        ref_r = model(s1, e1, f1)
    outs = sess.run(None, {"spec": s1.numpy(), "feat_erb": e1.numpy(),
                           "feat_spec": f1.numpy()})
    for i, name in enumerate(["enh", "mask", "lsnr", "df_alpha"]):
        d = np.abs(outs[i] - ref_r[i].numpy()).max()
        print(f"onnx vs torch {name:9s}: max|diff| {d:.3e}")
        assert d < 1e-4, f"ONNX differs from torch on {name}"

    # (3) end-to-end time domain through Rust iSTFT
    y_ref = torch.as_tensor(
        df_state.synthesis(as_complex(torch.from_numpy(ref_r[0].numpy()).squeeze(1)).numpy())
    ).numpy()
    y_onnx = torch.as_tensor(
        df_state.synthesis(as_complex(torch.from_numpy(outs[0]).squeeze(1)).numpy())
    ).numpy()
    d4 = np.abs(y_ref - y_onnx).max()
    rms = float(np.sqrt((y_ref ** 2).mean()))
    print(f"time-domain max|diff| {d4:.3e}  (ref rms {rms:.4f})")
    assert d4 < 1e-3

    # quick RTF (one call = 30 new frames = 100 ms of audio)
    t0 = time.perf_counter()
    for _ in range(5):
        sess.run(["enh"], {"spec": s1.numpy(), "feat_erb": e1.numpy(),
                           "feat_spec": f1.numpy()})
    dt = (time.perf_counter() - t0) / 5
    print(f"ONNX RTF (100 ms audio per call): {0.1/dt:.1f}")
    print("DF2 ONNX GATE: PASS")


if __name__ == "__main__":
    main()
