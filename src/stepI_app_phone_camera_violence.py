from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
import tensorflow as tf
from collections import deque

app = Flask(__name__)

MODEL_PATH = "action_violence_model.h5"
IMG_SIZE = 96

model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded")

prev_gray = None
history = deque(maxlen=3)   

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>Edge AI – Violence Detection</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <style>
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
      color: white;
      text-align: center;
    }

    .container {
      padding: 16px;
    }

    h1 {
      margin-top: 10px;
      font-size: 26px;
    }

    h2 {
      font-size: 16px;
      font-weight: normal;
      opacity: 0.8;
      margin-bottom: 12px;
    }

    video {
      width: 100%;
      max-width: 420px;
      border-radius: 14px;
      border: 2px solid rgba(255,255,255,0.2);
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    .status {
      margin-top: 15px;
      padding: 12px;
      border-radius: 10px;
      font-size: 22px;
      font-weight: bold;
      transition: all 0.3s ease;
    }

    .safe {
      background: rgba(0, 200, 0, 0.2);
      color: #00ff99;
      box-shadow: 0 0 15px rgba(0,255,150,0.4);
    }

    .danger {
      background: rgba(200, 0, 0, 0.25);
      color: #ff4d4d;
      box-shadow: 0 0 20px rgba(255,0,0,0.6);
      animation: pulse 1s infinite;
    }

    @keyframes pulse {
      0% { box-shadow: 0 0 10px rgba(255,0,0,0.4); }
      50% { box-shadow: 0 0 25px rgba(255,0,0,0.9); }
      100% { box-shadow: 0 0 10px rgba(255,0,0,0.4); }
    }

    .footer {
      margin-top: 14px;
      font-size: 12px;
      opacity: 0.7;
    }
  </style>
</head>

<body>
<div class="container">

  <h1>Edge AI – Violence Detection</h1>
  <h2>Mobile Camera · Real-Time Analysis</h2>

  <video id="video" autoplay playsinline></video>

  <div id="status" class="status safe">Initializing…</div>

  <div class="footer">
    AI runs on Edge Server · Camera on Device
  </div>

</div>

<script>
const video = document.getElementById("video");
const statusBox = document.getElementById("status");

navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
  video.srcObject = stream;
})
.catch(() => {
  statusBox.innerText = "Camera access denied";
  statusBox.className = "status danger";
});

const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

setInterval(() => {
  if (video.videoWidth === 0) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);

  canvas.toBlob(blob => {
    const formData = new FormData();
    formData.append("frame", blob);

    fetch("/predict", { method: "POST", body: formData })
      .then(res => res.json())
      .then(data => {
        statusBox.innerText = data.label;

        if (data.label === "VIOLENCE") {
          statusBox.className = "status danger";
        } else {
          statusBox.className = "status safe";
        }
      });
  }, "image/jpeg", 0.8);
}, 250);
</script>

</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    global prev_gray

    file = request.files["frame"]
    frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    if prev_gray is None:
        prev_gray = gray
        history.clear()
        return jsonify(label="NON-VIOLENCE")

    motion = cv2.absdiff(gray, prev_gray)
    prev_gray = gray

    motion = motion.astype("float32") / 255.0
    motion = np.expand_dims(motion, axis=(0, -1))

    preds = model.predict(motion, verbose=0)
    class_id = int(np.argmax(preds))

    history.append(class_id)

    
    if list(history) == [1, 1, 1]:
        label = "VIOLENCE"
    else:
        label = "NON-VIOLENCE"

    return jsonify(label=label)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
