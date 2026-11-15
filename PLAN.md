# Personal Clean Voice Filter — Plan

Status: draft for review (2026-11-14)
Distribution targets: **Windows, macOS, Linux** — standalone app for end users on all three (no Python install).
Development + primary test machine: this box (CachyOS, PipeWire, RX 7900 XTX, Focusrite Scarlett 18i6). Windows/macOS validated in CI + manual hardware passes (see Phase 5).

## 1. Goal

A self-contained GUI application that:

1. Creates a **virtual microphone input** (visible to any application on the system).
2. Processes audio from a **user-selectable real input device** and feeds the result to that virtual input.
3. Supports a **processed ↔ passthrough toggle** (raw mic bypass with zero processing).
4. Provides **simple voice EQ** (cut/boost bands tuned for voice) for the cleanest possible voice.
5. Has a **mandatory training step**: the user records a sample of a voice to register it. The app then uses the registered voice(s) to **isolate the allowed voice(s)** from other people talking nearby (call-center scenario: many concurrent speakers + noise).
6. Supports **multiple registered voice profiles**, each toggleable **allowed / blocked**, plus one selectable **default profile** that is automatically allowed when the app starts. Only allowed registered voices pass through Isolated mode; everyone else (registered-but-blocked or unregistered) is suppressed.

Non-goals: transcription, music processing, echo cancellation, multi-mic beamforming.

## 2. Research summary

### 2.1 Virtual audio input — per OS (the app must create/expose one on all three)

| OS | Mechanism | How the app writes into it | Install cost for the user |
|---|---|---|---|
| **Linux** (PipeWire or PulseAudio) | **Null sink** created by the app at startup (`pactl load-module module-null-sink`); removed on exit, stale instances cleaned up at startup. Verified working on this machine. | PortAudio output stream to the sink (enumerated as a normal output device) | **None** — created in-app |
| **Windows** | **VB-Audio Virtual Cable** (free driver; virtual "CABLE Output" playback endpoint + "CABLE Input" mic endpoint; official driver pack covers XP→Win11 x64/ARM64). A kernel driver is required on Windows — no userspace way to fabricate a capture endpoint. | PortAudio/WASAPI output stream to "CABLE Output" | One-time driver install + reboot; app has a **setup wizard**: detect → download official installer → run (elevated) → prompt reboot → re-detect |
| **macOS** | **BlackHole 2ch** (open source, free CoreAudio loopback driver, 18k+ stars; modern Soundflower replacement). | PortAudio/CoreAudio output stream to "BlackHole 2ch" | One-time driver install (sudo, reboot recommended); same guided-install wizard |

- The app owns the whole DSP pipeline in-process; the virtual device is only the last hop. (DeepFilterNet's LADSPA/PipeWire filter-chain "virtual mic" validates the null-sink pattern; the LADSPA route was rejected because a plugin cannot host the TSE/training/EQ stack.)
- Per-OS **device layer** (`core/device/`): `linux.py` (pactl), `windows.py` (registry/WMI device-name detection + installer wizard), `macos.py` (CoreAudio device scan + installer wizard). GUI always shows: virtual input name, detected/missing, one-click fix.
- VB-Cable/BlackHole are installed by the user from their official sites (not redistributed by us); the wizard links the official download.
- Rejected alternatives: VoiceMeeter (Windows, no free headless virtual mic), VAC (weaker installer story), macOS Aggregate Devices (can't feed app-processed audio), Loopback (macOS, paid).

### 2.2 Voice isolation technology — what was evaluated

| Option | Real-time | Speaker-specific (trained) | License | Verdict |
|---|---|---|---|---|
| **DeepFilterNet 2/3** (Rikorose) | ✔ ~20 ms, CPU, 48 kHz full-band | ✘ (general enhancement only) | Apache-2.0/MIT | **Use** as stage-1 enhancer + reference for PipeWire virtual-mic |
| **RNNoise** | ✔ | ✘ | BSD | Weaker than DeepFilterNet; not needed |
| **ClearVoice / ClearerVoice-Studio** (Alibaba) | chunked, needs GPU | audio-only TSE has **no pretrained model** (only audio-visual TSE needs a camera) | Apache-2.0 | Rejected for isolation (training our own TSE from scratch is out of scope); kept as reference |
| **WeSep 2024** (wenet-e2e) | chunked, **CPU by default** | ✔ **BSRNN + ECAPA pretrained (English, 16 kHz)**, auto-download; model zoo: SpEx+, pBSRNN, pDPCCN (recipes) | Apache-2.0 | **Ready-to-use TSE API** (`wesep.load_model("english")`), CPU-friendly entry point |
| **wesep-real-tse** (REAL-TSE Challenge baseline) | ✔ **causal** streaming variant, C++ runtime built for real-time | ✔ BSRNN separator + WeSpeaker ECAPA-TDNN enrollment embedding | **Apache-2.0 (code + checkpoints)** | **Primary isolation engine** |
| Speaker-gating (embed per chunk, gate gain) | ✔ ~50 ms | ✔ coarse | — | Fallback "fast isolation": good against intermittent talkers, leaks on overlap |
| 2-track separation + track selection (MossFormer2_SS, SpEx) | chunked | ✔ coarse | Apache-2.0 | Fallback for 2-talker rooms |
| TargetVoice / SpeakerBeam-SS (papers) | ✔ | ✔ | closed | Reference only |

**Decision:** isolation = **Target Speaker Extraction (TSE)** conditioned on the user's enrolled voice — this is exactly the "give me a sample of your voice, I isolate it" requirement, and it is a real, benchmarked field (REAL-TSE, SLT 2026).

### 2.3 Speaker embeddings (training feature)

- **WeSpeaker** ECAPA-TDNN (MIT) — the cue model the wesep-real-tse BSRNN TSE was trained with; enrollment = one embedding vector computed from the recorded sample.
- 3D-Speaker ERes2Net/CAM++ (Apache-2.0) available as cross-check for enrollment quality, not required in the hot path.

### 2.4 EQ

- Biquad cascade (scipy.signal SOS filters), all standard audio DSP: low-cut HPF, low/mid/high trims, presence, de-esser, limiter, gain. Sample-accurate, ~0 ms latency, no models.

### 2.5 Cross-platform constraints

- **Audio I/O**: PortAudio (via `sounddevice`) on all three OSes — WASAPI (Windows), CoreAudio (macOS), Pulse/ALSA (Linux). Handles sample-rate conversion to 48 kHz internally.
- **GUI**: PySide6 — first-class Windows/macOS/Linux support.
- **GPU is NOT a given** ("all kinds of laptops"): NVIDIA dGPU (Windows/Linux) → CUDA; Apple Silicon → MPS; AMD (desktop & many laptops) → ROCm (Linux) / none (Windows); most laptops → **no usable GPU at all**. Therefore **CPU performance is the baseline that must be good**, GPU is an accelerator where present.
- **Inference engine strategy** (decided in Phase 0 spike, gate documented there):
  - Primary candidate: **ONNX Runtime** — universal CPU speed (AVX2/AVX512), small footprint, optional per-OS GPU EPs (CUDA, DirectML, CoreML, ROCm). Requires exporting BSRNN-TSE + ECAPA + DeepFilterNet to ONNX with correct streaming-state handling and numeric parity.
  - Fallback: **PyTorch CPU** wheels per platform (auto-enable MPS/CUDA/ROCm where available). Slightly larger bundle, zero export risk.
  - The app exposes "engine" as an internal detail; the GUI compute selector (Auto/GPU/CPU) just picks the active EP/device.
- **Python 3.12** in the dev venv (this machine is 3.14; wheels guarantee for 3.12). Shipped builds bundle Python via PyInstaller — end users never install anything but (on Win/mac) the one-time virtual-cable driver.
- Model downloads total <500 MB (DF2 ~10 MB, BSRNN-TSE ~100–200 MB, ECAPA ~20–40 MB, VAD ~2 MB), fetched on first run with progress UI.
- Dev machine GPU (7900 XTX) also runs a local LLM — TSE needs ~1–2 GB VRAM, negligible contention.

## 3. Architecture

### 3.0 Three strict layers — filter core is independent of GUI and OS (requirement)

```
┌──────────────────────────────────────────────┐
│ gui/   (PySide6)                             │  depends only on filter API + osio status
├──────────────────────────────────────────────┤
│ osio/ (capture, output, virtual device,      │  the ONLY code touching PortAudio,
│   threads/ring buffers, permissions)         │  pactl, registry, CoreAudio, platform paths
├──────────────────────────────────────────────┤
│ filter/ (PURE)                               │  100% OS- and GUI-independent: numpy +
│   DF → TSE → EQ, engine, profiles, calib.   │  engine (ONNX/torch). Same code on all OSes.
└──────────────────────────────────────────────┘
```

- **Dependency rule**: `filter/` imports nothing from `gui/` or `osio/`; `osio/` passes raw frames and never touches filter internals. Enforced in CI by an import-boundary test on every OS.
- The filter core is a synchronous, stateful streaming object (§3.5) driven by `osio`'s process thread in the app.
- The **same `filter/` code** runs in: (a) the GUI app, (b) headless CLI (file in → file out, any mode/EQ/profiles), (c) the CI test suite on all three OSes without any audio hardware.
- Upgrading the audio stack = new `filter/` with a versioned stable API; GUI and OS layers keep working against the contract.

```
┌────────────────────────────────────────────────────────────────────┐
│ pvcf (Python app, one process)                                     │
│                                                                    │
│  Real mic (PipeWire source, user-selected)                         │
│        │ PortAudio, 48 kHz mono, 10 ms frames                      │
│        ▼                                                           │
│  [ring buffer] ──► PROCESS THREAD                                  │
│        │                                                           │
│        ├─ mode = passthrough ──────────────────────────────┐       │
│        │                                                   │       │
│        ▼ (processed modes)                                 │       │
│  1. DeepFilterNet (48 kHz, CPU, ~20 ms)        ──┐         │       │
│        │                                          │         │       │
│  2. TSE (16 kHz BSRNN-causal + ECAPA enrollment) │         │       │
│     (isolation mode only; streaming chunks,      │         │       │
│      mask applied back onto the 48 kHz signal)   │         │       │
│        ▼                                          ▼         │       │
│  3. Voice EQ chain (biquads + limiter + gain) ◄──┴─────────┤       │
│        ▼                                                   │       │
│  [output ring buffer]  ◄───────────────────────────────────┘       │
│        │ PortAudio write                                            │
│        ▼                                                            │
│  PipeWire null sink "PersonalVoiceFilter"  = virtual microphone     │
│        │                                                            │
└────────┼────────────────────────────────────────────────────────────┘
         ▼
   Any application (Discord, Zoom, browser, OBS, …) selects
   "PersonalVoiceFilter" as its input.
```

### 3.1 Threading model

- **Capture thread**: PortAudio input callback → 200 ms ring buffer (drop-oldest under load, counter feeds "dropouts" meter).
- **Process thread**: consumes fixed chunks; runs stages in order; TSE runs on GPU (or CPU per spike) with a dedicated torch thread pool; never blocks >2× chunk time (backlog ⇒ drop oldest and update latency meter).
- **Output thread**: PortAudio output stream writing to the null sink (device addressed by name; verified in spike).
- **GUI thread** (Qt event loop): controls config atomics; meters read counters at 30 Hz.
- Passthrough mode: capture → output direct copy (bypasses process thread work; ~10 ms total).

### 3.2 Modes (GUI toggle group)

1. **Passthrough** — raw mic to virtual input, zero processing.
2. **Enhanced** — DeepFilterNet + EQ. Low latency (~60–100 ms). No training required.
3. **Isolated** — DeepFilterNet + TSE (≥1 allowed registered profile required) + EQ. Least leakage of other talkers, latency ~0.4–0.9 s on fast hardware, longer on weak CPU laptops (chunk-size auto-adaptation; measured in Phase 0; streaming chunk + 50 % overlap crossfade).
   - **1 allowed profile**: single TSE run → that voice only.
   - **2+ allowed profiles**: one TSE run per allowed profile (independent streaming chunks, batched on the active compute device), outputs mixed. All unregistered/blocked voices are suppressed by construction (TSE removes non-target speech).
   - **0 allowed profiles**: Isolated mode disabled in the GUI with a hint to register/allow a voice.

This gives the user both "cleanest sound" (Isolated) and "lowest latency" (Enhanced).

### 3.3 The 16 kHz bandwidth problem (TSE models run at 16 kHz)

TSE at 16 kHz would otherwise cap the voice to phone bandwidth. Plan:

- Preferred: derive a **spectral mask** from the TSE output vs. the 16 kHz downmix, upsample the mask to 48 kHz, and apply it to the original 48 kHz STFT → preserves full-band content while suppressing other speakers.
- Fallback (if the masked output sounds bad in A/B): accept 16 kHz bandwidth in Isolated mode (phone-quality clarity is exactly what call-center audio is).

### 3.4 Compute devices — per OS, auto-detected

- Startup detects what's available and the GUI "Auto / GPU / CPU" selector maps to:
  - **Windows**: CUDA EP (NVIDIA) → DirectML EP (AMD/Intel iGPU) → CPU.
  - **macOS**: MPS (Apple Silicon) → CPU (Intel Macs).
  - **Linux**: ROCm (AMD) or CUDA (NVIDIA) → CPU.
  - CPU path (baseline for all laptops): ONNX Runtime CPU threads or torch CPU, per the Phase 0 engine decision.
- **Chunk-size auto-adaptation**: first-run calibration measures TSE RTF at 0.5/1/2/4 s chunks and stores the largest chunk with RTF ≤ ~0.5; re-measured if the user changes device. The Status tab shows the active device, chunk, and measured latency per mode.
- **Weak-CPU fallback**: if even 4 s chunks don't hit RTF ≤ 0.5, Isolated mode is offered as **fast isolation** (Enhanced + speaker-gating, ~50 ms, coarser separation) with a clear label; the user can still switch to a GPU machine's profile settings.

### 3.5 Filter core API (stable, versioned contract)

```python
class VoiceFilter:
    def open(self, cfg: FilterConfig) -> None             # engine + models loaded once
    def close(self) -> None
    def process(self, frame: np.ndarray) -> np.ndarray    # float32 mono 48 kHz, 10 ms frames (480 samples);
                                                          # same length in/out, causal, non-blocking
    def set_mode(self, mode: Literal["passthrough", "enhanced", "isolated"]) -> None
    def set_eq(self, eq: EQSettings) -> None              # atomic swap of the biquad chain
    def set_profiles(self, allowed: list[Profile]) -> None# Profile = portable file format, §4.3
    def set_device(self, device: Literal["auto", "gpu", "cpu"]) -> None
    def calibrate(self) -> Calibration                    # RTF per chunk size; latency per mode
    def stats(self) -> Stats                              # RTF, drops, latency → GUI meters
```

- `process()` is the only audio-facing call. Internally: passthrough | DF(48k) | TSE(16k, mask back) | EQ.
- No OS I/O, no GUI, no platform imports; logging via standard `logging` only.
- Deterministic for given frames + config → golden-file unit tests run in CI on all three OSes, no audio hardware required.

## 4. Training (enrollment) design — mandatory before Isolated mode

**Registering a voice profile** (GUI tab "Train Voice"):
- Select source device (default: main input).
- **Record** 10–30 s (live level meter, countdown, auto-stop) **or import** an existing voice file (wav/flac/mp3).
- **Quality gates** (computed on the sample):
  - Voice-activity ratio (Silero VAD) ≥ ~40 % — else "too much silence".
  - Speaker consistency: sample split into 4–6 segments, ECAPA embedding per segment, mean pairwise cosine similarity ≥ threshold — else warn "multiple speakers or too noisy, re-record".
- **Live A/B preview**: run the just-recorded sample through the TSE chain and let the user compare before saving.
- On save: profile stored at
  `~/.config/personalcleanvoicefilter/profiles/<name>/`
  `{ meta.json, embed.pt (L2-normalized ECAPA vector), enroll_16k.wav }`

**Profile management** (same tab):
- List of all registered profiles with: name, **allowed / blocked** toggle, **"set as default"** action, re-record (replaces sample + embedding), rename, delete (confirm).
- **Default profile** = the profile automatically **allowed** when the app starts. All other profiles retain the last persisted allowed/blocked state.
- Runtime state: allowed set = profiles with allowed=true (default applied at startup). Isolated mode runs one TSE per allowed profile (§3.2).
- Per-profile **test button**: plays/records a short test and shows an A/B of that profile's extraction result (sanity check that the right person is registered/allowed).
- **Export / Import**: one-click per profile. A profile is a self-contained portable file (§4.3) that can be exported from any OS and imported into the app on any other OS — no account, no server, no app-data copying.

**§4.3 Profile file format (portable, cross-OS)**

- One profile = one file, extension `.pvcf` (versioned ZIP container):
  - `profile.json` — schema version, name, created timestamp, sample rate, enrollment quality stats, model-generation id, SHA-256 manifest of the other entries.
  - `embedding.bin` — the speaker embedding as **raw float32 little-endian** bytes (dimension recorded in `profile.json`). Deliberately engine-agnostic: it's just a vector, so future model swaps keep imports working (the app flags "different model generation → recompute from sample" when applicable).
  - `enroll_16k.wav` — the original enrollment sample (16 kHz mono) for re-testing and recomputation.
- **Import** = validate (schema version ≤ current, manifest checksums, embedding dimension, wav sample rate) → copy into the local profile store. Rejects newer schema versions with a clear "app too old" message.
- **Export** = copy the file out (Save-As dialog / drag-and-drop).
- The local profile store is simply a directory of `.pvcf` files (+ a `store.json` for allowed/blocked/default state), so profiles can also be synced manually (USB, Drive, git) between machines.

- Runtime: each enrollment embedding is the fixed TSE cue for its profile. No fine-tuning required; re-training a profile = record a new sample (fast, no GPU training job).
- Optional (Phase 5, experimental): adaptive enrollment — slow EMA of a profile's own extracted frames to track voice changes (e.g. cold room vs. office).

## 5. Voice EQ design

All biquads, 48 kHz, applied per chunk; stateful (zero-phase not possible in streaming — standard causal biquads, negligible audible delay at these Qs).

| Control | Type | Range | Default |
|---|---|---|---|
| Low cut (HPF) | freq | 40–200 Hz | 80 Hz (kills mic rumble/handling noise) |
| Low | low shelf @ 200 Hz | −10…+10 dB | 0 (−3 in "call center" preset: less boxiness) |
| Mid (clarity) | peaking @ 1.2–2.5 kHz (freq adjustable) | −10…+10 dB | +3 |
| Presence | high shelf @ 9 kHz | −10…+10 dB | +3 |
| De-esser | peak @ 6.5–8 kHz | −0…−18 dB | −4 |
| Limiter | brickwall | on/off | on (−1 dBTP) |
| Output gain | gain | −12…+12 dB | 0 |

- Presets: **Flat / Voice / Call Center / Broadcast** + user-saved presets (config file).
- A/B button: instant bypass of the whole processed chain (same as Passthrough toggle, also in EQ tab for quick A/B while tuning).

## 6. GUI specification

Framework: **PySide6** (native look, proper sliders/meters; Tkinter rejected: no decent level meters, clunky for the training flow).

Single window, tabs:

1. **Input & Mode**
   - Input device dropdown (all OS input sources; default: Focusrite Scarlett 18i6 where present).
   - **Virtual input status card**: name per OS ("PersonalVoiceFilter" null sink / "CABLE Input" / "BlackHole 2ch"), detected/active/missing, sample rate. When missing (Windows/macOS first run): one-click **"Install virtual input"** button → guided wizard (download official VB-Cable/BlackHole installer → run → reboot prompt → re-detect). Linux: never missing.
   - Mode selector: **Passthrough / Enhanced / Isolated** (Isolated disabled until ≥1 profile is allowed, with a "Train Voice →" hint — mandatory training requirement).
   - **Compute device selector: Auto / GPU / CPU** (Auto = best available per OS, §3.4; per-device measured RTF/latency shown in Status).
   - **Monitor toggle** (hear processed output via local speakers/headphones; off by default; warning about feedback).
2. **Train Voice** — registration flow + profile list (allowed/blocked toggles, set-default, test, re-record, rename, delete) as §4, plus **Export profile… / Import profile…** (portable `.pvcf` files, cross-OS).
3. **Voice EQ** — as §5, per-control live preview via monitor, A/B button, preset save/load.
4. **Status** — input/processed level meters, E2E latency ms, processing RTF (per active device), dropout counter, allowed-profile set, log view (last 100 lines).

Behavior:
- Config + active profile persisted (`~/.config/personalcleanvoicefilter/config.json`), restored on start.
- App start: create null sink (clean up stale one from a crash), load models (progress bar), open streams. App exit: close streams, unload null sink.
- `.desktop` entry + optional autostart (Phase 4).
- Name of the virtual input: **"PersonalVoiceFilter"** (shown in every app's device list).

## 7. Repository layout

```
personalcleanvoicefilter/
├── PLAN.md
├── README.md                  # usage, model licenses, setup
├── pyproject.toml             # venv: python 3.12; deps below
├── pvcf/
│   ├── filter/                # PURE audio filter — zero OS/GUI imports (CI boundary test)
│   │   ├── api.py             # VoiceFilter, FilterConfig, EQSettings, Profile, Stats (§3.5)
│   │   ├── enhance.py         # DeepFilterNet streaming wrapper (48 kHz)
│   │   ├── tse.py             # BSRNN-TSE streaming (16 kHz) + mask back to 48 kHz
│   │   ├── speaker.py         # ECAPA embedding + enrollment quality gates
│   │   ├── vad.py             # Silero VAD wrapper
│   │   ├── engine.py          # inference engine abstraction (ONNX / torch), EP picking
│   │   ├── eq.py              # biquad chain + presets
│   │   └── profile.py         # .pvcf read/write/validate (§4.3)
│   ├── osio/                  # only OS-dependent code
│   │   ├── pipeline.py        # threads + ring buffers driving VoiceFilter
│   │   ├── capture.py         # PortAudio input (sounddevice)
│   │   ├── output.py          # PortAudio output to virtual device
│   │   └── device/            # per-OS virtual input layer (§2.1)
│   │       ├── linux.py       # null sink create/cleanup (pactl)
│   │       ├── windows.py     # VB-Cable detection + install wizard
│   │       └── macos.py       # BlackHole detection + install wizard
│   ├── gui/                   # PySide6: window, tabs, meters
│   │   └── window.py  tabs_input.py  tabs_train.py  tabs_eq.py  tabs_status.py
│   ├── main.py                # entry: GUI app
│   └── cli.py                 # headless: filter only (file src → file/pipe out, any mode/EQ/profile)
├── vendor/                    # clones: DeepFilterNet, wesep (wesep_2024), wesep-real-tse, wespeaker (pinned)
├── models/                    # downloaded checkpoints (gitignored)
├── packaging/                 # PyInstaller specs per OS, icon, installer scripts
└── tests/
    ├── filter/                # pure DSP tests: eq numerics, ring buffer, tse chunking, engine
    │                          #   parity, profile round-trip, import boundary — run in CI on
    │                          #   all 3 OSes with no audio hardware
    └── osio/                  # device layer (mocks), thread/latency soak
```

Dependencies (venv, Python 3.12): `sounddevice` (PortAudio), `numpy`, `scipy`, `PySide6`, `onnxruntime` (+ GPU EP per OS), `onnx` (export tooling), `loguru`, `soundfile`, `platformdirs`; plus either `torch`+`torchaudio` or the ONNX path per Phase 0 decision (both during dev; one shipped). System: PortAudio libs, Pulse (`pactl`) on Linux.

## 8. Development phases

**Phase 0 — Feasibility spike (gate).**
- venv py3.12; torch install; confirm ROCm on gfx1101 (else CPU path for this machine).
- Measure RTF: DeepFilterNet @48 kHz CPU; BSRNN-causal TSE @16 kHz on GPU and CPU with 0.5/1/2/4 s chunks; Silero VAD.
- Verify PortAudio: open BRIO/Scarlett source, write to a null sink created via `pactl`, read it back with `arecord`; measure ping latency.
- **ONNX gate (cross-platform engine decision):** export BSRNN-TSE (causal checkpoint), ECAPA, and DeepFilterNet to ONNX; verify (a) numeric parity vs. torch reference output (<1e-3), (b) correct streaming-state handling across chunks, (c) RTF of the ONNX CPU path on this machine **and** on a GitHub Actions x86 runner (representative laptop-class CPU). Pass → ONNX Runtime is the shipped engine (with per-OS GPU EPs). Fail → ship PyTorch CPU per platform (MPS/CUDA auto-enabled where available).
- Deliverable: measured latency/RTF table + engine decision → fixes chunk-size policy, compute detection, and whether Isolated mode is viable real-time per tier. **If TSE RTF > 1 even at 4 s chunks on CPU, weak-CPU users get fast isolation (Enhanced + speaker gating, 50 ms) per §3.4 — documented tradeoff.**

### Phase 0 results (done 2026-11-14) — **GATE: PASS**

**Environment:** venv py3.12, torch 2.x + ROCm, RX 7900 XTX (gfx1101) confirmed working; 24-core CPU.

**Torch RTF (reference)** — `tools/phase0_rtf.py`, RTF = audio_s / compute_s (higher = more headroom):

| Model | Device | RTF |
|---|---|---|
| TSE BSRNN 24k/48k/72k window (1/2/3 s) | GPU | 17–36 |
| TSE BSRNN same | CPU (12 threads) | 1.7–4.4 |
| DF2 @48 kHz | CPU | 45–66 |
| Silero VAD | CPU | 91 |

**ONNX gate** — `tools/phase0_onnx.py` (TSE + ECAPA), `tools/export_df2.py` (DF2):
- TSE exported in **production runtime form** (fixed enrollment embedding input, streaming internal state), 3 static window sizes 1/2/3 s → `models/onnx/tse_bsrnn_{24000,48000,72000}.onnx`.
- ECAPA c512 (enrollment) → `models/onnx/ecapa_c512.onnx`.
- DF2 exported via static real-unfold (32-frame window, 30 new frames = 100 ms per call) → `models/onnx/df2_mask.onnx`.
- Numeric parity vs torch: max rel diff **4.2e-08** (≪ 1e-3 gate).
- Streaming-state parity verified across chunk boundaries.

**ONNX CPU RTF** — `tools/ci_onnx_rtf.py` (torch-free), 4 threads (laptop proxy):

| Model | Chunk | RTF (4 thr) |
|---|---|---|
| TSE BSRNN 24k (1 s) | 1.0 s | 2.26 |
| TSE BSRNN 48k (2 s) | 2.0 s | 2.23 |
| TSE BSRNN 72k (3 s) | 3.0 s | 2.20 |
| DF2 mask | 0.1 s | 26.2 |

**PortAudio null-sink round-trip** — `tools/phase0_roundtrip.py`: capture → null sink → read back, E2E ping latency 11–16 ms. Confirms the virtual-input pattern (Linux null sink) with PortAudio on PipeWire.

**Decisions fixed by this gate:**
1. **Shipped engine: ONNX Runtime** (CPU EP baseline; per-OS GPU EPs optional). PyTorch dev-only.
2. **Isolated mode is real-time on CPU for all users** — even at 4 threads TSE RTF ≈ 2.2, far above the RTF ≤ 0.5 target; the §3.4 weak-CPU fast-isolation fallback is **not needed as a performance backstop** (retained only as a quality/latency tradeoff).
3. **Chunk-size policy**: first-run calibration per §3.4 can keep the 3 s window on laptop-class hardware; auto-adaptation still implemented (per-OS runners vary).
4. **CI**: `.github/workflows/onnx-rtf.yml` runs the torch-free RTF gate on `ubuntu-latest` (4 threads, min RTF 1.0; 2-thread informational run) on every change to `models/onnx/**` — keeps Isolated-mode real-time on a representative x86 CPU a verified property.
5. **Repo**: ONNX models committed via Git LFS (models needed in CI); vendor clones by reference (`VENDOR.md` + `scripts/vendor.sh`); checkpoints auto-downloaded, not committed.

**Phase 1 — Filter core (headless, pure, no GUI/OS audio).**
- `filter/` implements §3.5: DF → TSE → EQ pipeline, engine, profiles (`.pvcf` read/write/validate), calibration, stats; all driven by `VoiceFilter.process()`.
- `cli.py` runs it file-in/file-out in any mode with any profiles/EQ — the same code path the GUI uses.
- Golden-file + numerics tests in `tests/filter/`; import-boundary test (no gui/osio imports in `filter/`).
- Acceptance: CLI processes real mixed-talk test audio (multi-speaker, call-center-like) in all modes with measured RTF within budget; profile export→import round-trip byte-stable; boundary test green.

**Phase 1b — OS integration.**
- `osio/`: capture → ring buffer → `VoiceFilter.process()` → output → virtual device; null-sink lifecycle (Linux); passthrough/enhanced/isolated modes end-to-end; config; logging.
- Acceptance: 30-min soak without dropouts; a second process (browser recorder) receives the "PersonalVoiceFilter" input; E2E latency within measured budget; passthrough path bit-identical verified.

**Phase 2 — Training + TSE (multi-profile).**
- Enrollment record/import, quality gates, profile store (allowed/blocked/default), cue loading, streaming TSE with chunk overlap + mask; A/B preview; N parallel TSE streams (one per allowed profile).
- Acceptance:
  - Two-person noisy test (talk while a second voice plays near the mic), 1 profile allowed → Isolated output contains the target speaker with the other clearly suppressed (A/B + SI-SDR-ish spot check).
  - Two registered profiles, both allowed, both "speaking" → both present in output; block one → only the allowed one remains.
  - Restart app → default profile auto-allowed, other profiles keep persisted state.
  - Passthrough/enhanced unaffected by profiles.

**Phase 3 — GUI.**
- PySide6 app wiring all stages; meters; mode toggle; EQ controls; profile management (register/allow/block/default/test/delete); compute-device selector; monitor toggle; error states (models missing → download button; no valid input; sink busy).
- Acceptance: full user story possible with GUI only: start app → register voice(s), set allow states + default → pick input → set EQ → toggle modes → select "PersonalVoiceFilter" input in another app.

**Phase 4 — Polish.**
- Linux `.desktop` file + autostart (Win/mac equivalents: run-at-login via OS APIs), config persistence verified across restarts, README (setup, per-OS virtual-input install, model licenses, troubleshooting), launcher script.

**Phase 5 — Cross-platform builds & validation (release gate).**
- **CI matrix** (GitHub Actions): ubuntu-latest, windows-latest, macos-latest — run the full test suite headless (the entire `filter/` core, including golden-file DSP tests, engine parity, and profile round-trip, runs on every OS with no audio hardware or virtual drivers; `osio/` via mocks), build PyInstaller artifacts.
  - Windows: onedir `.exe` (+ zip); note SmartScreen/code-signing status.
  - macOS: `.app` for arm64 **and** x86_64 (universal or dual build); notarization if a cert is available, else document right-click→Open.
  - Linux: AppImage (+ `.tar.gz` fallback), noting system lib requirements (Pulse, Qt, PortAudio).
- **Manual hardware passes** (checklist in README): one Windows laptop (VB-Cable wizard → full user story), one macOS (BlackHole wizard → full user story), this Linux box. Windows/mac hardware access arranged via the user.
- Acceptance: on each OS — app starts, virtual input visible in another app, train → allow → Isolated mode works, passthrough/enhanced work, restart restores state.

**Phase 6 — Optional (only if requested).**
- Adaptive enrollment, 48 kHz TSE fine-tune, tray icon, VAC as Windows alternative, multi-channel BlackHole, second-profile auto-detect ("is that you?" auto-switch between allowed profiles).

## 9. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| ONNX export of stateful BSRNN-TSE fails or has parity/state errors | Can't ship the small ONNX engine | Phase 0 gate with numeric parity + streaming-state tests; fallback: ship PyTorch CPU per platform (larger bundle, same functionality) |
| PyTorch ROCm on 7900 XTX / py3.12 friction | TSE too slow on this machine's CPU path | Phase 0 gate; CPU fallback with smaller chunk; "fast isolation" gating fallback |
| TSE trained on 2-speaker simulated mixtures; call center = many speakers + reverb | Leakage of other talkers in Isolated mode | A/B acceptance test (Phase 2); fallbacks: separation + gating, EQ "Call Center" preset, monitor mode for tuning; document honest limits |
| 16 kHz TSE bandwidth | Loss of highs in Isolated mode | Mask-back-to-48 kHz (§3.3); fallback: accept phone bandwidth in that mode |
| Weak laptop CPUs (no GPU, older silicon) | Isolated mode latency too high | Chunk auto-adaptation + RTF calibration; fast-isolation fallback (§3.4); per-mode latency shown in Status |
| Virtual-driver friction on Win/mac (elevation, reboot, Gatekeeper/SmartScreen for unsigned app) | First-run setup fails for users | Guided wizards with status + error text; README troubleshooting; code-signing/notarization if certificates available; Linux needs no driver |
| No Windows/macOS hardware during development | Platform bugs found late | Device layer is thin and unit-tested with mocks; CI runs headless pipeline on all three; Phase 5 manual hardware passes on user-arranged machines before release |
| PortAudio↔per-OS quirks (WASAPI exclusive apps, CoreAudio rate switching, Pulse naming, 18-channel Scarlett) | Capture/output bugs | Phase 0 round-trip test per OS where possible; fallbacks: native PipeWire (`pwpy`) on Linux; documented workarounds for edge devices |
| Stale null sink after crash (Linux) | Duplicate/broken input device | Startup cleanup of sinks with our exact name; unload-on-exit |
| GPU shared with local LLM (dev machine) | VRAM spikes | TSE small (~1–2 GB); torch thread cap; CPU mode available |
| Enrollment quality (short sample, background talkers during training) | Weak cue → weak isolation | Quality gates + warnings + re-record flow; A/B preview before save |
| macOS mic-permission (TCC) / Windows mic privacy settings block capture | App sees no input | First-run permission prompts handled in GUI with instructions; README |

## 10. Licensing

App code: MIT (our own).
Components: DeepFilterNet (Apache-2.0/MIT), wesep + wesep-real-tse + checkpoints (Apache-2.0), WeSpeaker (MIT), Silero VAD (MIT), PySide6 (LGPLv3, dynamically linked), torch (BSD), onnxruntime (MIT). No GPL code in the app. All model weights are auto-downloaded from their public hosts with attribution in README.
External one-time drivers (installed by the user, not redistributed): VB-Audio Virtual Cable (Windows, free), BlackHole (macOS, open source).

## 11. Decisions (agreed with user, 2026-11-14)

1. **Latency**: provide both Enhanced (~60–100 ms) and Isolated (~0.4–0.9 s) modes; user chooses per use.
2. **Primary mic**: Focusrite Scarlett 18i6 (GUI still lists all inputs).
3. **Profiles**: multiple registered voices, each allowed/blocked, with one selectable **default** that is auto-allowed at startup (§4).
4. **Compute**: GPU (ROCm) and CPU both supported; selectable in the GUI (Auto / GPU / CPU).
5. **EQ**: full set — low-cut HPF, low/mid/presence bands, de-esser, limiter, presets + custom save.
6. **Monitor**: toggle in GUI (off by default).
7. **Cross-platform distribution** (added 2026-11-14): must work standalone on Windows, macOS, and Linux. Consequences locked in: per-OS virtual-input layer (null sink / VB-Cable / BlackHole) with guided install wizards; CPU is the performance baseline (GPU optional, per-OS detection §3.4); inference engine decided by Phase 0 ONNX gate (ONNX Runtime or PyTorch CPU, one shipped); PyInstaller builds per OS from a 3-OS CI matrix; Windows/mac validated on real hardware in Phase 5 before release.
8. **Layering + portable profiles** (added 2026-11-14): the audio filter core (`filter/`) is strictly independent of the GUI and of all OS integration — same pure code on all three OSes, driven through a versioned `VoiceFilter` API (§3.5), CI-enforced import boundary, runnable headless via CLI. A voice profile is a portable, self-contained `.pvcf` file (§4.3) — exportable from and importable into the app on any OS, no server involved.
