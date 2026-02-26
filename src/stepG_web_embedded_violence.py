from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import requests
from tensorflow.keras.models import load_model
import time
from datetime import datetime


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

body {
    margin: 0;
    background: #0f2027;
    font-family: Arial, Helvetica, sans-serif;
    color: white;
    text-align: center;
}

.header {
    padding: 20px 10px 10px 10px;
}

h1 {
    margin: 0;
    font-size: 24px;
    font-weight: 600;
}

.subtitle {
    font-size: 14px;
    color: #bbb;
    margin-top: 5px;
}

.video-container {
    display: flex;
    justify-content: center;
    margin-top: 20px;
}

.video-box {
    width: 640px;              /* Match ESP32 resolution */
    max-width: 95%;
    border-radius: 10px;
    overflow: hidden;
    border: 2px solid #333;
    box-shadow: 0 0 20px rgba(0,0,0,0.6);
    background: black;
}

.video-box img {
    width: 100%;
    height: auto;
    display: block;
    object-fit: contain;       /* Prevent zoom/stretch */
}

.status-bar {
    margin-top: 15px;
    font-size: 14px;
}

.status-dot {
    height: 10px;
    width: 10px;
    background-color: #00ff00;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}

.footer {
    margin-top: 20px;
    font-size: 12px;
    color: #888;
}

.time {
    margin-top: 5px;
    font-size: 13px;
    color: #aaa;
}

</style>
</head>

<body>

<div class="header">
    <h1>Edge-AI Violence Detection</h1>
    <div class="subtitle">
        Real-Time Monitoring | MobileNetV2 | Flask Web Server
    </div>
</div>

<div class="video-container">
    <div class="video-box">
        <img src="/video">
    </div>
</div>

<div class="status-bar">
    <span class="status-dot"></span>
    System Running on Local Edge Node
</div>

<div class="time">
    Current Time: <span id="clock"></span>
</div>

<div class="footer">
    ESP32-CAM → Edge Inference (Laptop) → Browser Client
</div>

<script>
function updateClock() {
    const now = new Date();
    document.getElementById("clock").innerHTML = now.toLocaleString();
}
setInterval(updateClock, 1000);
updateClock();
</script>

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
            response = requests.get(ESP32_CAM_URL, timeout=5)
            if response.status_code != 200:
                continue

            img_array = np.frombuffer(response.content, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            
            prediction = model.predict(preprocess(frame), verbose=0)[0][0]

            if prediction > THRESHOLD:
                label = f"VIOLENCE ({prediction:.2f})"
                color = (0, 0, 255)
            else:
                label = f"NON-VIOLENCE ({1 - prediction:.2f})"
                color = (0, 255, 0)

            font = cv2.FONT_HERSHEY_SIMPLEX

           
            cv2.putText(frame, label, (10, 25),
                        font, 0.6, color, 2, cv2.LINE_AA)

            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp,
                        (10, frame.shape[0] - 12),
                        font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.2)

        except Exception as e:
            print("Stream error:", e)
            time.sleep(1)
            continue


@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict', methods=['POST'])
def predict_dummy():
    return {"message": "Push API not used in this architecture"}, 200


if __name__ == "__main__":
    print("[INFO] Edge-AI Flask server running on port 5000")
    app.run(host='0.0.0.0', port=5000, debug=False)