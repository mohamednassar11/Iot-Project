# Iot-Project# IoT Smoke/Fire Detection with AI (Logistic Regression)

This project reads live sensor data on Raspberry Pi (MQ-2 via MCP3008, DS18B20 temperature, and a flame sensor), applies rule-based thresholds, and adds an AI probability score (Logistic Regression) to improve alarm decisions.

## Repository Structure

- `main_with_ai.py` : Runtime script (sensors + AI inference + buzzer)
- `lr_model.joblib` : Trained Logistic Regression model
- `requirements.txt` : Python dependencies
- `README_RUN.txt` : Quick run steps (optional)

Optional for real-data training:
- `collect_data.py` : Collects labeled sensor data to CSV
- `train_lr_real.py` : Trains a new `lr_model.joblib` from real data

## Hardware

- Raspberry Pi (SPI + 1-Wire enabled)
- MQ-2 gas/smoke sensor connected through MCP3008 (ADC)
- DS18B20 temperature sensor (1-Wire)
- Flame sensor (digital)
- Buzzer

Default BCM pins used in `main_with_ai.py`:
- Flame: GPIO 17
- Buzzer: GPIO 27
- MCP3008 CS: D8
- MCP3008 channel: P0

## Alarm Logic

The alarm is triggered if either rule-based or AI-based conditions are met:

- Rules alarm:
  - Flame detected OR
  - MQ-2 average > `SMOKE_THRESHOLD` OR
  - Temperature > `TEMP_THRESHOLD`
- AI alarm:
  - `risk >= AI_THRESHOLD`

Final:
`ALARM = RulesAlarm OR AIAlarm`

## AI Feature Set

The Logistic Regression model expects 7 features in this exact order:

1. `temp_c`
2. `mq2_avg`
3. `mq2_voltage`
4. `flame` (0/1)
5. `mq2_ratio = mq2_avg / BASELINE`
6. `temp_delta`
7. `mq2_delta`

`BASELINE` is computed during MQ-2 calibration in clean air.

## Raspberry Pi Setup

### Enable SPI and 1-Wire (one time)

```bash
sudo raspi-config
# Interface Options -> SPI -> Enable
# Interface Options -> 1-Wire -> Enable
sudo reboot
