// Log-mel features. Mirrors train/features.py exactly.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_SAMPLE_RATE 16000
#define KWS_CLIP_SAMPLES 16000
#define KWS_FRAME_LEN 480
#define KWS_FRAME_HOP 320
#define KWS_NFFT 512
#define KWS_NUM_MEL 32
#define KWS_NUM_FRAMES (1 + (KWS_CLIP_SAMPLES - KWS_FRAME_LEN) / KWS_FRAME_HOP)  // 49

// Call once at start. Returns 0 on success, -1 if out of memory.
int features_init(void);

// samples: KWS_CLIP_SAMPLES int16 values (oldest first)
// out:     KWS_NUM_FRAMES * KWS_NUM_MEL floats, row-major [frame][mel]
void features_compute(const int16_t *samples, float *out);

#ifdef __cplusplus
}
#endif
