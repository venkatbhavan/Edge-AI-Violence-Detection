from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import requests
from tensorflow.keras.models import load_model
import time

# ===============================
# CONFIG
# ===============================
ESP32_CAM_URL = "http://192.168.0.137/capture"
MODEL_PATH = "action_violence_model.h5"
IMG_SIZE = 96
THRESHOLD = 0.45

app = Flask(__name__)
model = load_model(MODEL_PATH)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Edge-AI Violence Detection</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body { background: #111; color: white; text-align: center; font-family: Arial; }
h2 { margin-top: 10px; }
img { width: 95%; border: 3px solid #444; }
</style>
</head>
<body>
<h2>Edge-AI Violence Detection</h2>
<img src="/video">
<p>ESP32-CAM → Edge AI → Mobile Device</p>
</body>
</html>
"""

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray = gray / 255.0
    gray = np.expand_dims(gray, axis=(0, -1))
    return gray

def generate_frames():
    while True:
        try:
            r = requests.get(ESP32_CAM_URL, timeout=10)
            img = np.frombuffer(r.content, np.uint8)
            frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            pred = model.predict(preprocess(frame), verbose=0)[0][0]

            if pred > THRESHOLD:
                label = f"VIOLENCE ({pred:.2f})"
                color = (0, 0, 255)
            else:
                label = f"NON-VIOLENCE ({1-pred:.2f})"
                color = (0, 255, 0)

            cv2.putText(frame, label, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.25)

        except:
            continue

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("[INFO] Embedded Edge-AI server running")
    app.run(host='0.0.0.0', port=5000)
