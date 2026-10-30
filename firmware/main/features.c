// Log-mel features in plain C. Mirrors train/features.py exactly.
// No ESP-only calls here, so tests/test_parity.py can compile it on a laptop.
#include "features.h"

#include <math.h>
#include <stdlib.h>

#define NBINS (KWS_NFFT / 2 + 1)
#define FMIN 80.0
#define FMAX 7600.0
#define EPS 1e-6f
#define LOG_OFFSET 8.0f
#define LOG_SCALE 4.0f

static float s_win[KWS_FRAME_LEN];
static float s_cos[KWS_NFFT / 2];
static float s_sin[KWS_NFFT / 2];
static uint16_t s_bitrev[KWS_NFFT];
static float s_fb[KWS_NUM_MEL][NBINS];   // ~4 KB, fine on ESP32 DRAM
static float s_re[KWS_NFFT];
static float s_im[KWS_NFFT];
static int s_ready = 0;

static double hz_to_mel(double f) { return 2595.0 * log10(1.0 + f / 700.0); }
static double mel_to_hz(double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); }

int features_init(void) {
    if (s_ready) return 0;
    const double pi = 3.14159265358979323846;

    for (int i = 0; i < KWS_FRAME_LEN; i++)
        s_win[i] = (float)(0.5 * (1.0 - cos(2.0 * pi * i / KWS_FRAME_LEN)));

    for (int i = 0; i < KWS_NFFT / 2; i++) {
        s_cos[i] = (float)cos(2.0 * pi * i / KWS_NFFT);
        s_sin[i] = (float)(-sin(2.0 * pi * i / KWS_NFFT));
    }

    int bits = 0;
    while ((1 << bits) < KWS_NFFT) bits++;
    for (int i = 0; i < KWS_NFFT; i++) {
        int r = 0;
        for (int b = 0; b < bits; b++)
            if (i & (1 << b)) r |= 1 << (bits - 1 - b);
        s_bitrev[i] = (uint16_t)r;
    }

    double mels[KWS_NUM_MEL + 2];
    double lo_m = hz_to_mel(FMIN), hi_m = hz_to_mel(FMAX);
    for (int i = 0; i < KWS_NUM_MEL + 2; i++)
        mels[i] = mel_to_hz(lo_m + (hi_m - lo_m) * i / (KWS_NUM_MEL + 1));
    for (int m = 0; m < KWS_NUM_MEL; m++) {
        double lo = mels[m], mid = mels[m + 1], hi = mels[m + 2];
        for (int k = 0; k < NBINS; k++) {
            double f = (double)k * KWS_SAMPLE_RATE / KWS_NFFT;
            double up = (f - lo) / (mid - lo);
            double down = (hi - f) / (hi - mid);
            double v = up < down ? up : down;
            s_fb[m][k] = (float)(v > 0.0 ? v : 0.0);
        }
    }
    s_ready = 1;
    return 0;
}

// In-place radix-2 FFT on s_re / s_im (length KWS_NFFT)
static void fft(void) {
    for (int i = 0; i < KWS_NFFT; i++) {
        int j = s_bitrev[i];
        if (j > i) {
            float t = s_re[i]; s_re[i] = s_re[j]; s_re[j] = t;
            t = s_im[i]; s_im[i] = s_im[j]; s_im[j] = t;
        }
    }
    for (int len = 2; len <= KWS_NFFT; len <<= 1) {
        int half = len >> 1;
        int step = KWS_NFFT / len;
        for (int i = 0; i < KWS_NFFT; i += len) {
            for (int k = 0; k < half; k++) {
                float wr = s_cos[k * step], wi = s_sin[k * step];
                int a = i + k, b = a + half;
                float tr = s_re[b] * wr - s_im[b] * wi;
                float ti = s_re[b] * wi + s_im[b] * wr;
                s_re[b] = s_re[a] - tr; s_im[b] = s_im[a] - ti;
                s_re[a] += tr;          s_im[a] += ti;
            }
        }
    }
}

void features_compute(const int16_t *samples, float *out) {
    for (int f = 0; f < KWS_NUM_FRAMES; f++) {
        const int16_t *x = samples + f * KWS_FRAME_HOP;
        for (int i = 0; i < KWS_FRAME_LEN; i++)
            s_re[i] = ((float)x[i] / 32768.0f) * s_win[i];
        for (int i = KWS_FRAME_LEN; i < KWS_NFFT; i++) s_re[i] = 0.0f;
        for (int i = 0; i < KWS_NFFT; i++) s_im[i] = 0.0f;
        fft();
        for (int k = 0; k < NBINS; k++) s_re[k] = s_re[k] * s_re[k] + s_im[k] * s_im[k];
        for (int m = 0; m < KWS_NUM_MEL; m++) {
            float acc = 0.0f;
            const float *w = s_fb[m];
            for (int k = 0; k < NBINS; k++) acc += w[k] * s_re[k];
            out[f * KWS_NUM_MEL + m] = (logf(acc + EPS) + LOG_OFFSET) / LOG_SCALE;
        }
    }
}
