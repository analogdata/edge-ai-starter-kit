# Edge AI Roadmap: from this kit to real projects

You just ran a neural network on a chip that costs a few dollars. Here is the honest path from here. Do the stages in order. Each one builds on the last.

## Stage 0: Finish this kit (a weekend)
- Get "on" and "off" working on your own voice.
- Read `features.c` and `main.cc` line by line. They are short.
- Change the word list and retrain once.

## Stage 1: Embedded basics, properly (2 to 4 weeks)
- GPIO, timers, interrupts, I2C and SPI on the ESP32, using ESP-IDF (not Arduino).
- FreeRTOS: tasks, queues, why the audio loop must never block.
- Read a datasheet and wire a sensor from scratch.

*Goal: you can explain every line in `app_main()`.*

## Stage 2: Add gestures (2 to 3 weeks)
Voice is one sense. Motion is another.
- Add an MPU6050 or similar IMU over I2C.
- Collect your own data: wave, circle, shake, tap. 50+ samples each.
- Train a small 1D CNN on accelerometer windows, export int8, run it with the same TFLite Micro setup.

*Goal: one board that reacts to both your voice and your hand.*

## Stage 3: Learn the ML side properly (3 to 4 weeks)
- Why spectrograms work. What a convolution actually does.
- Overfitting, validation sets, confusion matrices.
- Quantisation: post-training vs quantisation-aware training. Measure the accuracy drop every time.
- Data matters more than model size. Record your own words, accents and rooms.

*Goal: when accuracy is bad, you know whether it is the data, the features or the model.*

## Stage 4: Make it production-grade (4 to 6 weeks)
- Memory and latency budgets. Profile before you optimise.
- Power: deep sleep, wake on sound, battery life maths.
- Over-the-air updates and a safe rollback.
- Test on real devices in real rooms: noise, distance, different people.

## Stage 5: Connect, only when it helps (2 to 4 weeks)
Edge AI does not mean never using the network. It means the network is optional.
- Decide on the device, send only results (for example "light turned on"), not raw audio.
- MQTT to a dashboard, or a cloud backend for fleet management.

## Stage 6: Ship a portfolio project
Pick one and finish it end to end: voice-controlled device, gesture remote, predictive-maintenance sensor, keyword-triggered recorder. Write it up with accuracy numbers, memory use and a demo video. That is what teams want to see.

---

Want a guided path with live sessions, code reviews and projects? See the EdgeAI Engineering Bootcamp at [analogdata.io](https://analogdata.io).
