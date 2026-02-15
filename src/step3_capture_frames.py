import cv2
import numpy as np
import requests

# 🔴 CHANGE THIS TO YOUR ESP32-CAM IP
ESP32_CAM_URL = "http://192.168.0.137/capture"

while True:
    try:
        response = requests.get(ESP32_CAM_URL, timeout=5)
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is not None:
            cv2.imshow("ESP32-CAM Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Error:", e)

cv2.destroyAllWindows()
