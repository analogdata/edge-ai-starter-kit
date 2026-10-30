"""
Data loading for the keyword model.

Two sources:
  * Google Speech Commands v2 (real voices, ~2.3 GB download, free licence CC BY 4.0)
  * --synthetic: fake tone-based "words". ONLY for checking the pipeline works. The model
    trained on it will not understand speech.
"""
import os
import tarfile
import urllib.request

import numpy as np
from scipy.io import wavfile

import features as F

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

# Words the board will understand. Edit this list, retrain, re-export. Done.
COMMANDS = ["yes", "no", "on", "off", "up", "down", "stop", "go"]
SILENCE, UNKNOWN = "_silence_", "_unknown_"
LABELS = COMMANDS + [SILENCE, UNKNOWN]


def download(data_dir):
    root = os.path.join(data_dir, "speech_commands_v0.02")
    if os.path.isdir(root) and os.path.exists(os.path.join(root, "validation_list.txt")):
        return root
    os.makedirs(root, exist_ok=True)
    tar_path = os.path.join(data_dir, "speech_commands_v0.02.tar.gz")
    if not os.path.exists(tar_path):
        print("Downloading Speech Commands v2 (~2.3 GB). This takes a while...")
        urllib.request.urlretrieve(DATA_URL, tar_path)
    print("Extracting...")
    with tarfile.open(tar_path) as t:
        t.extractall(root)
    return root


def _read(path):
    sr, x = wavfile.read(path)
    assert sr == F.SAMPLE_RATE
    x = x.astype(np.int16)
    out = np.zeros(F.CLIP_SAMPLES, dtype=np.int16)
    out[: min(len(x), F.CLIP_SAMPLES)] = x[: F.CLIP_SAMPLES]
    return out


def _shift(x, rng, max_ms=100):
    s = int(rng.integers(-max_ms, max_ms + 1) * F.SAMPLE_RATE / 1000)
    return np.roll(x, s) if s == 0 else (
        np.concatenate([x[s:], np.zeros(s, np.int16)]) if s > 0
        else np.concatenate([np.zeros(-s, np.int16), x[:s]]))


def _mix_noise(x, noises, rng, max_gain=0.15):
    n = noises[int(rng.integers(len(noises)))]
    start = int(rng.integers(0, len(n) - F.CLIP_SAMPLES))
    g = rng.uniform(0, max_gain)
    y = x.astype(np.float32) + g * n[start:start + F.CLIP_SAMPLES].astype(np.float32)
    return np.clip(y, -32768, 32767).astype(np.int16)


def load_speech_commands(data_dir, unknown_ratio=1.0, aug_copies=1, seed=0, max_per_class=None):
    rng = np.random.default_rng(seed)
    root = download(data_dir)
    val_set = set(open(os.path.join(root, "validation_list.txt")).read().split())
    test_set = set(open(os.path.join(root, "testing_list.txt")).read().split())

    noises = []
    nd = os.path.join(root, "_background_noise_")
    for f in os.listdir(nd):
        if f.endswith(".wav"):
            noises.append(wavfile.read(os.path.join(nd, f))[1].astype(np.int16))

    split = {"train": ([], []), "val": ([], []), "test": ([], [])}
    unknown = {"train": [], "val": [], "test": []}
    for word in sorted(os.listdir(root)):
        wdir = os.path.join(root, word)
        if not os.path.isdir(wdir) or word.startswith("_"):
            continue
        files = sorted(os.listdir(wdir))
        if word in COMMANDS and max_per_class:
            files = files[:max_per_class]
        for fn in files:
            rel = f"{word}/{fn}"
            s = "val" if rel in val_set else "test" if rel in test_set else "train"
            if word in COMMANDS:
                split[s][0].append(_read(os.path.join(wdir, fn)))
                split[s][1].append(LABELS.index(word))
            else:
                unknown[s].append(os.path.join(wdir, fn))

    out = {}
    for s in ("train", "val", "test"):
        xs, ys = split[s]
        per_class = max(1, len(xs) // len(COMMANDS))
        n_extra = int(per_class * unknown_ratio)
        pick = rng.permutation(len(unknown[s]))[:n_extra]
        for i in pick:
            xs.append(_read(unknown[s][i]))
            ys.append(LABELS.index(UNKNOWN))
        for _ in range(n_extra):   # silence = background noise at random volume
            n = noises[int(rng.integers(len(noises)))]
            st = int(rng.integers(0, len(n) - F.CLIP_SAMPLES))
            g = rng.uniform(0.0, 1.0)
            xs.append((g * n[st:st + F.CLIP_SAMPLES]).astype(np.int16))
            ys.append(LABELS.index(SILENCE))
        xs, ys = np.stack(xs), np.array(ys)
        if s == "train" and aug_copies > 0:
            ax, ay = [], []
            for _ in range(aug_copies):
                for x, y in zip(xs, ys):
                    ax.append(_mix_noise(_shift(x, rng), noises, rng))
                    ay.append(y)
            xs, ys = np.concatenate([xs, np.stack(ax)]), np.concatenate([ys, np.array(ay)])
        out[s] = (xs, ys)
        print(f"{s}: {len(xs)} clips")
    return out


def make_synthetic(n_per_class=150, seed=0):
    """Fake 'words': each class is a different tone sweep + noise. Pipeline test only."""
    rng = np.random.default_rng(seed)
    t = np.arange(F.CLIP_SAMPLES) / F.SAMPLE_RATE
    out = {}
    for s, n in (("train", n_per_class), ("val", n_per_class // 3), ("test", n_per_class // 3)):
        xs, ys = [], []
        for c, _ in enumerate(LABELS):
            for _ in range(n):
                f0 = 300 + 350 * c + rng.uniform(-40, 40)
                sweep = rng.uniform(-300, 300)
                env = np.exp(-((t - rng.uniform(0.4, 0.6)) ** 2) / 0.02)
                x = 9000 * env * np.sin(2 * np.pi * (f0 + sweep * t) * t) + rng.normal(0, 300, t.size)
                xs.append(np.clip(x, -32768, 32767).astype(np.int16))
                ys.append(c)
        idx = rng.permutation(len(xs))
        out[s] = (np.stack(xs)[idx], np.array(ys)[idx])
    return out
