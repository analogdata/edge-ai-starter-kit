// Pins and tuning. Change these to match your wiring.
#pragma once

// I2S microphone (INMP441 or similar)
#define PIN_I2S_BCLK 4    // INMP441 SCK
#define PIN_I2S_WS   5    // INMP441 WS
#define PIN_I2S_DIN  6    // INMP441 SD
// Mic L/R pin to GND => left channel (what the code reads)

#define PIN_LED 2         // plain LED on GPIO 2 (use 48 or 38 for the RGB LED on some S3 boards)

// INMP441 gives 24-bit audio inside a 32-bit slot. We shift it down to 16-bit.
// If the board never hears you, lower this (14 -> 12). If it always clips, raise it.
#define MIC_SHIFT 14

// A command counts only if the model is at least this sure (0.0 - 1.0)
#define CONFIDENCE 0.80f
// ...and the same word wins this many times in a row (we listen 4 times per second)
#define CONFIRMATIONS 2
// Ignore very quiet audio (saves power, fewer false triggers). RMS on the 16-bit scale.
#define MIN_RMS 120
// Arena for TensorFlow Lite Micro tensors. If the log says "AllocateTensors failed", raise it.
#define TENSOR_ARENA_BYTES (48 * 1024)
