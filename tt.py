#!/usr/bin/env python3
import threading
import time
import glob
import RPi.GPIO as GPIO
import busio
import board
from digitalio import DigitalInOut
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn
import serial
import numpy as np
from joblib import load
from flask import Flask, jsonify, render_template_string

# =========================
# CONFIG
# =========================
FLAME_PIN = 17
BUZZER_PIN = 27

SERIAL_PORT = "/dev/serial0"
BAUD_RATE = 9600
PHONE_NUMBER = "+201062948826"

TEMP_THRESHOLD = 50.0
SMOKE_FACTOR = 1.5
AI_THRESHOLD = 0.70

MODEL_PATH = "lr_model.joblib"

# ---- AI NORMALIZATION (MATCH TRAINING SCALE) ----
MAX_TEMP = 100.0
MAX_MQ2 = 60000.0

# =========================
# GPIO
# =========================
GPIO.setmode(GPIO.BCM)
GPIO.setup(FLAME_PIN, GPIO.IN)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)

system_running = True

# =========================
# DS18B20
# =========================
base_dir = "/sys/bus/w1/devices/"
device_folders = glob.glob(base_dir + "28-*")
device_file = device_folders[0] + "/temperature" if device_folders else None

def read_temp():
    if device_file is None:
        return 0.0
    with open(device_file, "r") as f:
        return int(f.read()) / 1000.0

# =========================
# MCP3008 (MQ-2)
# =========================
spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
cs = DigitalInOut(board.D8)
mcp = MCP.MCP3008(spi, cs)
mq2 = AnalogIn(mcp, MCP.P0)

# =========================
# SHARED DATA
# =========================
sensor_data = {
    "temperature": 0.0,
    "mq2_value": 0.0,
    "flame": False,
    "risk": 0.0,
    "alert": False
}

data_lock = threading.Lock()
SMOKE_READINGS = []
BASELINE = 0.0
SMOKE_THRESHOLD = 0.0

# =========================
# GSM
# =========================
try:
    gsm = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(5)
    gsm_ok = True
except:
    gsm_ok = False
    gsm = None

sms_sent = False
last_sms = 0
SMS_COOLDOWN = 30

def send_sms(msg):
    global sms_sent, last_sms
    if not gsm_ok:
        return
    if sms_sent and time.time() - last_sms < SMS_COOLDOWN:
        return

    gsm.write(b"AT\r")
    time.sleep(0.5)
    gsm.write(b"AT+CMGF=1\r")
    time.sleep(0.5)
    gsm.write(f'AT+CMGS="{PHONE_NUMBER}"\r'.encode())
    time.sleep(0.5)
    gsm.write(msg.encode())
    gsm.write(bytes([26]))
    time.sleep(3)

    sms_sent = True
    last_sms = time.time()
    print("📨 SMS sent")

# =========================
# MQ-2 CALIBRATION
# =========================
def calibrate_mq2():
    global BASELINE, SMOKE_THRESHOLD
    print("🔥 Warming MQ-2 (15 sec)...")
    time.sleep(15)
    values = []
    for _ in range(50):
        values.append(float(mq2.value))
        time.sleep(0.2)
    BASELINE = sum(values) / len(values)
    SMOKE_THRESHOLD = BASELINE * SMOKE_FACTOR
    print(f"Baseline={BASELINE:.0f} Threshold={SMOKE_THRESHOLD:.0f}")

# =========================
# LOAD AI MODEL
# =========================
try:
    lr_model = load(MODEL_PATH)
    print("✅ AI model loaded")
except:
    lr_model = None
    print("⚠️ AI model not found")

# =========================
# SENSOR THREADS (UNCHANGED)
# =========================
def flame_thread():
    while True:
        if system_running:
            with data_lock:
                sensor_data["flame"] = GPIO.input(FLAME_PIN) == GPIO.HIGH
        time.sleep(0.2)

def mq2_thread():
    while True:
        if system_running:
            SMOKE_READINGS.append(float(mq2.value))
            if len(SMOKE_READINGS) > 5:
                SMOKE_READINGS.pop(0)
            with data_lock:
                sensor_data["mq2_value"] = sum(SMOKE_READINGS) / len(SMOKE_READINGS)
        time.sleep(0.2)

def temp_thread():
    while True:
        if system_running:
            with data_lock:
                sensor_data["temperature"] = read_temp()
        time.sleep(0.5)

# =========================
# AI THREAD (FIXED)
# =========================
def ai_thread():
    last_t, last_s = None, None
    while True:
        if lr_model is None:
            time.sleep(1)
            continue

        with data_lock:
            t = sensor_data["temperature"]
            s = sensor_data["mq2_value"]
            f = 1.0 if sensor_data["flame"] else 0.0

        ratio = s / BASELINE if BASELINE > 0 else 0.0
        dt = 0.0 if last_t is None else t - last_t
        ds = 0.0 if last_s is None else s - last_s

        last_t, last_s = t, s

        # ---- NORMALIZATION FIX ----
        t_n = t / MAX_TEMP
        s_n = s / MAX_MQ2
        dt_n = dt / MAX_TEMP
        ds_n = ds / MAX_MQ2

        X = np.array([[t_n, s_n, f, ratio, dt_n, ds_n]], dtype=float)
        risk = float(lr_model.predict_proba(X)[0][1])

        with data_lock:
            sensor_data["risk"] = risk

        time.sleep(0.5)

# =========================
# ALERT THREAD
# =========================
def alert_thread():
    global sms_sent
    while True:
        with data_lock:
            rule_alarm = (
                sensor_data["flame"] or
                sensor_data["mq2_value"] > SMOKE_THRESHOLD or
                sensor_data["temperature"] > TEMP_THRESHOLD
            )
            ai_alarm = sensor_data["risk"] >= AI_THRESHOLD
            sensor_data["alert"] = rule_alarm or ai_alarm

        GPIO.output(BUZZER_PIN, GPIO.HIGH if sensor_data["alert"] else GPIO.LOW)

        if sensor_data["alert"]:
            send_sms(
                f"🔥 FIRE ALERT!\n"
                f"Temp: {sensor_data['temperature']:.1f}C\n"
                f"Smoke: {int(sensor_data['mq2_value'])}\n"
                f"Risk: {sensor_data['risk']:.2f}"
            )
        else:
            sms_sent = False

        time.sleep(1)

# =========================
# FLASK WEB APP
# =========================
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Fire Monitoring</title>
<style>
body{background:#0f172a;color:#e5e7eb;font-family:Arial;text-align:center}
.card{background:#020617;padding:25px;border-radius:12px;width:360px;margin:auto}
.safe{color:#22c55e}
.danger{color:#ef4444}
</style>
<script>
setInterval(()=>{
fetch('/api').then(r=>r.json()).then(d=>{
document.getElementById('t').innerText=d.temperature.toFixed(1)
document.getElementById('s').innerText=Math.round(d.mq2_value)
document.getElementById('f').innerText=d.flame?'YES':'NO'
document.getElementById('r').innerText=d.risk.toFixed(2)
let a=document.getElementById('a')
a.innerText=d.alert?'🔥 DANGER':'✅ SAFE'
a.className=d.alert?'danger':'safe'
})
},1000)
</script>
</head>
<body>
<h1>🔥 Fire Monitoring</h1>
<div class="card">
<p>🌡 Temp: <span id="t">0</span> °C</p>
<p>💨 Smoke: <span id="s">0</span></p>
<p>🔥 Flame: <span id="f">NO</span></p>
<p>🤖 AI Risk: <span id="r">0</span></p>
<h2>Status: <span id="a" class="safe">SAFE</span></h2>
</div>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api")
def api():
    with data_lock:
        return jsonify({
            "temperature": float(sensor_data["temperature"]),
            "mq2_value": float(sensor_data["mq2_value"]),
            "flame": bool(sensor_data["flame"]),
            "risk": float(sensor_data["risk"]),
            "alert": bool(sensor_data["alert"])
        })

def web_thread():
    app.run(host="0.0.0.0", port=5000)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    try:
        calibrate_mq2()
        for t in [
            flame_thread,
            mq2_thread,
            temp_thread,
            ai_thread,
            alert_thread,
            web_thread
        ]:
            threading.Thread(target=t, daemon=True).start()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        GPIO.cleanup()
        print("Exiting...")
