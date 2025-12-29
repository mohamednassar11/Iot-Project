import threading
import time
import glob
import RPi.GPIO as GPIO
import busio
import board
from digitalio import DigitalInOut
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn

from joblib import load
import numpy as np

# ---------------- Pin Setup ----------------
FLAME_PIN = 17
BUZZER_PIN = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(FLAME_PIN, GPIO.IN)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# ---------------- DS18B20 Setup ----------------
base_dir = '/sys/bus/w1/devices/'
device_folders = glob.glob(base_dir + '28-*')
if not device_folders:
    # لو DS18B20 مش متوصل/1-Wire مش شغال هيتشاف ده على الراسبيري
    device_file = None
else:
    device_file = device_folders[0] + '/temperature'

def read_temp():
    """Read DS18B20 temperature in Celsius."""
    if device_file is None:
        return 0.0
    with open(device_file, 'r') as f:
        temp = int(f.read())
    return temp / 1000.0

# ---------------- MCP3008 Setup (MQ-2) ----------------
spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
cs = DigitalInOut(board.D8)  # CS pin
mcp = MCP.MCP3008(spi, cs)
mq2_channel = AnalogIn(mcp, MCP.P0)

# ---------------- MQ-2 Calibration ----------------
SMOKE_READINGS = []
BASELINE = 0.0
SMOKE_THRESHOLD = 0.0

def calibrate_mq2(duration=60):
    """Calibrate MQ-2 sensor baseline in clean air (seconds)."""
    global BASELINE, SMOKE_THRESHOLD
    print("🔥 Calibrating MQ-2... keep sensor in clean air.")
    readings = []
    start_time = time.time()
    while time.time() - start_time < duration:
        readings.append(float(mq2_channel.value))
        time.sleep(0.2)

    if not readings:
        BASELINE = 0.0
        SMOKE_THRESHOLD = 0.0
        print("⚠️ MQ-2 calibration failed (no readings).")
        return

    BASELINE = sum(readings) / len(readings)
    SMOKE_THRESHOLD = BASELINE * 1.5  # 50% above baseline
    print(f"✅ Calibration done. Baseline={BASELINE:.0f}, Smoke Threshold={SMOKE_THRESHOLD:.0f}")

# ---------------- Temperature Threshold ----------------
TEMP_THRESHOLD = 50.0  # °C

# ---------------- AI Config ----------------
MODEL_PATH = "lr_model.joblib"
AI_THRESHOLD = 0.70  # ارفعها لو إنذارات كتير غلط (0.80/0.85)

lr_model = None

# ---------------- Shared Sensor Data ----------------
sensor_data = {
    "temperature": 0.0,
    "mq2_value": 0.0,
    "mq2_voltage": 0.0,
    "flame": False,
    "risk": 0.0,
}

# ---------------- Sensor Threads ----------------
def flame_thread():
    while True:
        # بعض حساسات اللهب بتطلع 0 عند وجود لهب و 1 عند عدم وجوده
        # لو لاحظت العكس في الواقع، اعكس السطر ده.
        sensor_data["flame"] = bool(GPIO.input(FLAME_PIN))
        time.sleep(0.2)

def mq2_thread():
    while True:
        SMOKE_READINGS.append(float(mq2_channel.value))
        if len(SMOKE_READINGS) > 5:  # Moving average window
            SMOKE_READINGS.pop(0)

        sensor_data["mq2_value"] = sum(SMOKE_READINGS) / len(SMOKE_READINGS)
        sensor_data["mq2_voltage"] = float(mq2_channel.voltage)
        time.sleep(0.2)

def temp_thread():
    while True:
        sensor_data["temperature"] = float(read_temp())
        time.sleep(0.5)

# ---------------- AI Thread ----------------
def ai_thread():
    """
    IMPORTANT: feature order MUST match training:
      [temp_c, mq2_avg, mq2_voltage, flame_int, mq2_ratio, temp_delta, mq2_delta]
    """
    global lr_model

    last_temp = None
    last_mq2 = None

    while True:
        temp_c = float(sensor_data["temperature"])
        mq2_avg = float(sensor_data["mq2_value"])
        mq2_v = float(sensor_data["mq2_voltage"])
        flame_int = 1.0 if sensor_data["flame"] else 0.0

        mq2_ratio = (mq2_avg / BASELINE) if BASELINE and BASELINE > 0 else 0.0

        temp_delta = 0.0 if last_temp is None else (temp_c - last_temp)
        mq2_delta = 0.0 if last_mq2 is None else (mq2_avg - last_mq2)

        last_temp = temp_c
        last_mq2 = mq2_avg

        risk = 0.0
        if lr_model is not None:
            try:
                X = np.array([[temp_c, mq2_avg, mq2_v, flame_int, mq2_ratio, temp_delta, mq2_delta]], dtype=float)
                risk = float(lr_model.predict_proba(X)[0, 1])
            except Exception:
                risk = 0.0

        sensor_data["risk"] = risk
        time.sleep(0.5)

# ---------------- Display & Buzzer ----------------
def display_thread():
    last_print = ""
    while True:
        flame = sensor_data["flame"]
        smoke_value = sensor_data["mq2_value"]
        smoke_voltage = sensor_data["mq2_voltage"]
        temperature = sensor_data["temperature"]
        risk = sensor_data["risk"]

        flame_status = "🔥 Flame detected!" if flame else "✅ No flame"
        smoke_status = "⚠️ Smoke HIGH!" if (SMOKE_THRESHOLD > 0 and smoke_value > SMOKE_THRESHOLD) else "✅ Smoke Safe"
        temp_status = "🌡 Temp HIGH!" if temperature > TEMP_THRESHOLD else f"🌡 Temp: {temperature:.2f}°C"

        # Safety-first: rules + AI
        rule_alarm = (
            flame or
            (SMOKE_THRESHOLD > 0 and smoke_value > SMOKE_THRESHOLD) or
            (temperature > TEMP_THRESHOLD)
        )
        ai_alarm = (risk >= AI_THRESHOLD)

        if rule_alarm or ai_alarm:
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
        else:
            GPIO.output(BUZZER_PIN, GPIO.LOW)

        output = (
            f"{temp_status} | 💨 MQ-2: {smoke_value:.0f} ({smoke_voltage:.2f}V) | "
            f"{flame_status} | {smoke_status} | 🤖 Risk: {risk:.2f} | "
            f"ALARM: {'YES' if (rule_alarm or ai_alarm) else 'NO'}"
        )

        if output != last_print:
            print(output)
            last_print = output

        time.sleep(0.5)

# ---------------- Main ----------------
if __name__ == "__main__":
    try:
        calibrate_mq2(duration=60)

        # Load AI model
        try:
            lr_model = load(MODEL_PATH)
            print(f"✅ LR model loaded: {MODEL_PATH}")
        except Exception as e:
            lr_model = None
            print(f"⚠️ Could not load LR model ({MODEL_PATH}). AI disabled. Reason: {e}")

        # Start threads
        for func in [flame_thread, mq2_thread, temp_thread, ai_thread, display_thread]:
            t = threading.Thread(target=func, daemon=True)
            t.start()

        # Keep alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        GPIO.cleanup()
        print("Exiting...")
