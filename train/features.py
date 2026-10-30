"""
Audio features: log-mel spectrogram.

IMPORTANT: firmware/main/features.c does the SAME maths in C.
If you change a number here, change it there too (and rerun tests/test_parity.py).
"""
import numpy as np

SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000        # 1 second
FRAME_LEN = 480             # 30 ms
FRAME_HOP = 320             # 20 ms
NFFT = 512
NUM_MEL = 32
FMIN = 80.0
FMAX = 7600.0
EPS = 1e-6
# Fixed scaling so the model sees roughly -1.5 .. +1.5
LOG_OFFSET = 8.0
LOG_SCALE = 4.0
NUM_FRAMES = 1 + (CLIP_SAMPLES - FRAME_LEN) // FRAME_HOP   # 49


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank():
    """(NUM_MEL, NFFT//2+1) triangular filters, HTK mel scale."""
    nbins = NFFT // 2 + 1
    mels = np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), NUM_MEL + 2)
    hz = _mel_to_hz(mels)
    bin_hz = np.arange(nbins) * SAMPLE_RATE / NFFT
    fb = np.zeros((NUM_MEL, nbins), dtype=np.float64)
    for m in range(NUM_MEL):
        lo, mid, hi = hz[m], hz[m + 1], hz[m + 2]
        up = (bin_hz - lo) / (mid - lo)
        down = (hi - bin_hz) / (hi - mid)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    return fb


_FB = mel_filterbank()
_WIN = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(FRAME_LEN) / FRAME_LEN))


def log_mel(clips_int16):
    """
    clips_int16: (N, CLIP_SAMPLES) int16 (or (CLIP_SAMPLES,))
    returns:     (N, NUM_FRAMES, NUM_MEL) float32
    """
    x = np.asarray(clips_int16)
    single = x.ndim == 1
    if single:
        x = x[None, :]
    x = x.astype(np.float64) / 32768.0
    n = x.shape[0]
    idx = (np.arange(NUM_FRAMES)[:, None] * FRAME_HOP) + np.arange(FRAME_LEN)[None, :]
    frames = x[:, idx] * _WIN                       # (N, F, 480)
    spec = np.fft.rfft(frames, n=NFFT, axis=-1)     # zero-pads to 512
    power = spec.real ** 2 + spec.imag ** 2         # (N, F, 257)
    mel = power @ _FB.T                             # (N, F, 32)
    out = (np.log(mel + EPS) + LOG_OFFSET) / LOG_SCALE
    out = out.astype(np.float32)
    return out[0] if single else out
