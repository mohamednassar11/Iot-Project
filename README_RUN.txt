IoT Smoke/Fire Detection + AI (Logistic Regression)

Folder contents (must stay together):
- main_with_ai.py
- lr_model.joblib
- requirements.txt

Raspberry Pi one-time setup:
1) Enable SPI + 1-Wire:
   sudo raspi-config
   Interface Options -> SPI -> Enable
   Interface Options -> 1-Wire -> Enable
   sudo reboot

Run steps:
1) Copy this folder to the Pi home:
   /home/pi/iot_ai_smoke

2) Open terminal and run:
   cd ~/iot_ai_smoke
   sudo apt update
   sudo apt install -y python3-pip python3-venv
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

3) Run:
   python main_with_ai.py

Notes:
- MQ-2 is read through MCP3008 on channel P0, CS pin D8 (as in code).
- Flame sensor pin is GPIO 17 (BCM).
- Buzzer pin is GPIO 27 (BCM).
- AI alarm triggers when Risk >= AI_THRESHOLD inside main_with_ai.py (default 0.70).
  If too many false alarms, increase to 0.80 or 0.85.
- If flame logic is inverted (alarm with no flame), invert the FLAME read line in flame_thread().
