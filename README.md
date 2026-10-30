# EdgeAI Starter Kit: offline voice commands on an ESP32

A few-dollar chip. A tiny neural network. No internet, no cloud, no per-request fees.

Say **"on"** and the LED turns on. Say **"off"** and it turns off. The model runs on the ESP32 itself, and your voice never leaves the board.

```
 mic (I2S) ──► 1-second audio buffer ──► log-mel features ──► tiny int8 CNN ──► action
                                          (plain C, 49x32)     (~21 KB model)    (LED, relay, ...)
                          all of this runs on the ESP32, 4 times per second
```

## What you need

| Item | Notes | Rough cost (USD) |
|---|---|---|
| ESP32-S3 dev board | ESP32-S3 DevKitC-1 or similar | $6 - 10 |
| INMP441 I2S microphone | Or any I2S mic with the same pins | $2 - 5 |
| 5 jumper wires | Female-female or breadboard | ~$1 |
| Laptop | Windows, macOS or Linux, Python 3.10 - 3.13 for training | |
| ESP-IDF v6.1 | [Install guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/) | free |

### Tested versions

| Component | Version | Notes |
|---|---|---|
| ESP-IDF | v6.1 (6.1.0) | Needs v5.1 or newer for the TFLite component. `main/CMakeLists.txt` uses the v6 component names `esp_driver_gpio` and `esp_driver_i2s`; on v5.x, use `driver` instead |
| Toolchain | xtensa-esp-elf esp-15.2.0_20251204 | Installed by ESP-IDF |
| Target | ESP32-S3 | Set in `firmware/sdkconfig.defaults` |
| espressif/esp-tflite-micro | 1.4.1 | Managed component, pinned in `firmware/dependencies.lock` |
| espressif/esp-nn | 1.4.2 | Pulled in by esp-tflite-micro |
| Python (training) | 3.12 | TensorFlow does not support 3.14 yet |
| tensorflow | 2.21.0 | Includes keras 3.15.1 |
| numpy | 2.5.4 | |
| scipy | 1.18.1 | |
| pytest | 9.2.0 | Optional |
| CMake | 3.16 or newer | Minimum set in `firmware/CMakeLists.txt`; ESP-IDF tools install one |

`train/requirements.txt` lists only `tensorflow>=2.16`, `numpy` and `scipy` (unpinned). To reproduce the exact tested environment:

```bash
pip install tensorflow==2.21.0 numpy==2.5.4 scipy==1.18.1
```

## Wiring

| INMP441 | ESP32-S3 |
|---|---|
| VDD | 3V3 |
| GND | GND |
| L/R | GND (left channel) |
| SCK | GPIO 4 |
| WS | GPIO 5 |
| SD | GPIO 6 |

Different pins? Change them in `firmware/main/board_config.h`.

## Quick start (3 steps)

### 1. Train the model (on your laptop)

```bash
cd train
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

This downloads Google's Speech Commands dataset (about 2.3 GB, one time), trains a small CNN, shrinks it to 8-bit numbers (int8), and writes `model_data.cc` and `kws_config.h` into `firmware/main`. On a laptop CPU this takes a while, so start it and have a chai. The script prints the accuracy of the int8 model at the end, which is the one the ESP32 actually runs.

Want to check the whole pipeline in about a minute first? `python train.py --synthetic`. It uses fake tones, so the model will not understand speech. It only proves your setup works.

Short on time or disk? `python train.py --max-per-class 1500 --epochs 15`.

### 2. Build and flash

```bash
cd ../firmware
idf.py set-target esp32s3
idf.py build flash monitor
```

The first build downloads the `esp-tflite-micro` component automatically, so you need internet for that one step only.

### 3. Talk to it

Open the serial monitor. You should see `Model ready`. Now say one of: **yes, no, on, off, up, down, stop, go**. Say it clearly, about a hand's distance from the mic.

```
I (4210) edgeai: on (93% sure, NN ms per inference)   <- example output
```

## How it works (the 10th-grade version)

1. **Listen.** The microphone sends 16,000 numbers per second. We keep the last 1 second.
2. **Turn sound into a picture.** We slice the second into 49 short pieces and measure how much energy sits in 32 pitch bands (a log-mel spectrogram). Sound becomes a small 49 x 32 image. This is plain C in `features.c`.
3. **Ask the model.** A tiny convolutional network (the same family of ideas used to recognise photos) looks at that image and gives a score for each word.
4. **Act, but only if sure.** A word counts only if the score is at least 80% and it wins twice in a row. Then there is a 1-second cooldown so one "on" does not fire five times.

Why int8? Normal models use 32-bit decimals. Squeezing to 8-bit integers makes the model about 4 times smaller and much faster on a small chip, usually with a tiny accuracy drop. `train.py` measures that drop for you.

## Make it yours

- **Different words:** edit `COMMANDS` in `train/dataset.py`, retrain. Any word in the Speech Commands set works (yes, no, up, down, left, right, on, off, stop, go, plus numbers zero to nine and a few more).
- **Control real things:** edit `on_command()` in `firmware/main/main.cc`. Replace the LED with a relay, a motor driver or a servo.
- **Too sensitive or too deaf:** tune `CONFIDENCE`, `MIN_RMS` and `MIC_SHIFT` in `board_config.h`.

## Troubleshooting

| Problem | Fix |
|---|---|
| Build says `model_data.cc not found` | Run step 1 first. |
| `AllocateTensors failed` | Raise `TENSOR_ARENA_BYTES` in `board_config.h`. |
| Nothing is ever detected | Lower `MIC_SHIFT` (14 to 12), or `MIN_RMS`. Check that L/R is on GND. |
| Random false triggers | Raise `CONFIDENCE` to 0.9 or `CONFIRMATIONS` to 3. |
| Works on my voice only | Normal for a small model. Use `--aug-copies 2`, or record your own samples (see ROADMAP). |
| Different ESP32 chip (classic, C3) | Change `idf.py set-target`. The code is not chip-specific, but pins may differ. |

## Check the maths yourself

```bash
python tests/test_parity.py
```

This compiles the exact C feature code the ESP32 runs, feeds it test audio, and compares it against the Python feature code used for training. If they ever drift apart, your model will quietly get worse on the board, so run this whenever you touch `features.c` or `features.py`.

## What is inside

```
train/      Python: data, features, model, int8 export (writes into firmware/main)
firmware/   ESP-IDF project: I2S mic, features, TensorFlow Lite Micro inference
tests/      C-vs-Python feature check
ROADMAP.md  Where to go next: from this kit to real edge AI engineering
```

## Credits and licence

Dataset: Google Speech Commands v2 (CC BY 4.0). Inference: TensorFlow Lite Micro via Espressif's `esp-tflite-micro`. Code: MIT, see `LICENSE`.

Built by [Analog Data](https://analogdata.io). *From Bare Metal to Machine Intelligence.*
