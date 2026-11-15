"""Phase 0 spike: Pulse round-trip through a PipeWire null sink.

Creates a null sink (virtual mic), copies mic input -> sink output in real
time, records the sink's monitor source, and measures the E2E latency by
cross-correlating the two recordings.

Uses `soundcard` (direct PulseAudio API) because PortAudio on this system
uses the ALSA backend and cannot address the null sink by name.

Usage:
    .venv/bin/python tools/phase0_roundtrip.py [--mic NAME] [--secs 5]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time

import numpy as np

# soundcard 0.4.6 infers its program name from sys.argv[1] at import time
if len(sys.argv) < 2:
    sys.argv = [sys.argv[0], "pvcf"]
import soundcard as sc  # noqa: E402

SINK = "PersonalVoiceFilter"
SR = 48000
BLOCK = 480  # 10 ms


def create_sink() -> str:
    subprocess.run(
        ["pactl", "unload-module", "PersonalVoiceFilterSink"],
        capture_output=True,
    )
    out = subprocess.run(
        [
            "pactl", "load-module", "module-null-sink",
            f"sink_name={SINK}",
            f"sink_properties=device.description={SINK}",
        ],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"null sink load failed: {out.stderr}")
    lines = subprocess.run(["pactl", "list", "short", "modules"],
                           capture_output=True, text=True).stdout.splitlines()
    for l in lines:
        if "module-null-sink" in l and SINK in l:
            return l.split()[0]
    raise RuntimeError("module id not found")


def destroy_sink(module_id: str) -> None:
    subprocess.run(["pactl", "unload-module", module_id], capture_output=True)


def _find(name: str, pool):
    for d in pool:
        if d.name == name:
            return d
    return None


def find_speaker(name: str):
    for _ in range(30):  # poll up to 3 s: Pulse registration lags
        d = _find(name, sc.all_speakers())
        if d:
            return d
        time.sleep(0.1)
    raise RuntimeError(f"speaker {name!r} not found")


def find_source(name: str):
    """Exact-name source lookup. include_loopback so monitor sources
    ('Monitor of <sink>') are visible; exact match avoids fuzzy hits."""
    for _ in range(30):
        pool = sc.all_microphones(include_loopback=True, exclude_monitors=False)
        d = _find(name, pool)
        if d:
            return d
        time.sleep(0.1)
    raise RuntimeError(f"source {name!r} not found")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", default="BRIO Ultra HD Webcam Analog Stereo")
    ap.add_argument("--secs", type=float, default=5.0)
    args = ap.parse_args()

    module_id = create_sink()
    try:
        mic = find_source(args.mic)
        sink = find_speaker(SINK)
        monitor = find_source(f"Monitor of {SINK}")
        print(f"mic: {mic.name} | sink: {sink.name} | monitor: {monitor.name}")

        buf_in: list[np.ndarray] = []
        buf_mon: list[np.ndarray] = []
        stop_mon = threading.Event()

        with mic.recorder(samplerate=SR, channels=1, blocksize=BLOCK) as rec_in, \
             sink.player(samplerate=SR, channels=1, blocksize=BLOCK) as out, \
             monitor.recorder(samplerate=SR, channels=1, blocksize=BLOCK) as rec_mon:

            def mon_pump():
                while not stop_mon.is_set():
                    try:
                        buf_mon.append(rec_mon.record(None))
                    except Exception:
                        break

            t = threading.Thread(target=mon_pump, daemon=True)
            t.start()
            start = time.monotonic()
            while time.monotonic() - start < args.secs:
                data = rec_in.record(None)  # (frames, channels) float32
                buf_in.append(data[:, 0].copy())
                out.play(data)  # 1:1 copy, Pulse remaps channels

            stop_mon.set()
            t.join(timeout=2.0)

        mon = np.concatenate(buf_mon)[:, 0]
        insig = np.concatenate(buf_in)[:len(mon)]
        print(f"frames in: {len(insig)/SR:.2f}s  monitor: {len(mon)/SR:.2f}s")

        # cross-correlate over a 2 s overlap window
        n = min(len(insig), len(mon)) - 2 * SR
        a = insig[n // 2 : n // 2 + 2 * SR]
        b = mon[n // 2 : n // 2 + 2 * SR]
        a -= a.mean(); b -= b.mean()
        corr = np.correlate(b, a, mode="full")
        lag = int(np.argmax(corr) - (len(a) - 1))
        print(f"cross-correlation lag: {lag} samples = {lag / SR * 1000:.1f} ms")
        return 0
    finally:
        destroy_sink(module_id)


if __name__ == "__main__":
    sys.exit(main())
