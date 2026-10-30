"""
Checks that the C feature code (what runs on the ESP32) gives the same numbers
as the Python feature code (what the model was trained on).

Run from the repo root:   python tests/test_parity.py
Needs gcc and numpy. No ESP32 required.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "train"))
import features as F  # noqa: E402


def main():
    tmp = tempfile.mkdtemp()
    exe = os.path.join(tmp, "host_features")
    subprocess.check_call([
        "gcc", "-O2", "-o", exe,
        os.path.join(ROOT, "tests", "host_features.c"),
        os.path.join(ROOT, "firmware", "main", "features.c"),
        "-lm",
    ])

    rng = np.random.default_rng(0)
    t = np.arange(F.CLIP_SAMPLES) / F.SAMPLE_RATE
    clips = {
        "noise": (rng.normal(0, 3000, F.CLIP_SAMPLES)).astype(np.int16),
        "tone_800hz": (12000 * np.sin(2 * np.pi * 800 * t)).astype(np.int16),
        "sweep": (9000 * np.sin(2 * np.pi * (200 + 2500 * t) * t)).astype(np.int16),
        "silence": np.zeros(F.CLIP_SAMPLES, dtype=np.int16),
        "quiet_noise": (rng.normal(0, 20, F.CLIP_SAMPLES)).astype(np.int16),
    }

    worst = 0.0
    for name, x in clips.items():
        inp, outp = os.path.join(tmp, "in.raw"), os.path.join(tmp, "out.raw")
        x.tofile(inp)
        subprocess.check_call([exe, inp, outp])
        c = np.fromfile(outp, dtype=np.float32).reshape(F.NUM_FRAMES, F.NUM_MEL)
        p = F.log_mel(x)
        err = float(np.max(np.abs(c - p)))
        worst = max(worst, err)
        print(f"{name:12s} max |C - Python| = {err:.5f}")

    # Feature units are ~1.0 wide; 0.02 is well below what int8 quantisation can see.
    assert worst < 0.02, f"C and Python features disagree (worst {worst})"
    print("PASS: firmware features match training features")


if __name__ == "__main__":
    main()
