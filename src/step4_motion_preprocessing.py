import cv2
import numpy as np
import requests
import time

ESP32_CAM_URL = "http://192.168.0.137/capture"  # your IP

prev_gray = None

while True:
    try:
        response = requests.get(ESP32_CAM_URL, timeout=8)
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (96, 96))

        if prev_gray is None:
            prev_gray = gray
            time.sleep(0.3)
            continue

        motion = cv2.absdiff(gray, prev_gray)
        prev_gray = gray

        cv2.imshow("Original Frame", frame)
        cv2.imshow("Motion Image", motion)

        time.sleep(0.3)  # 🔑 THIS IS THE KEY FIX

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Error:", e)
        time.sleep(1)

cv2.destroyAllWindows()
