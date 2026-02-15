import cv2
import numpy as np
import requests
import time
import os

ESP32_CAM_URL = "http://192.168.0.137/capture"  # your ESP32 IP

SAVE_VIOLENT = "dataset/violent"
SAVE_NON_VIOLENT = "dataset/non_violent"

os.makedirs(SAVE_VIOLENT, exist_ok=True)
os.makedirs(SAVE_NON_VIOLENT, exist_ok=True)

prev_gray = None
count_v = 0
count_nv = 0

print("Press:")
print("  v → save VIOLENT action")
print("  n → save NON-VIOLENT action")
print("  q → quit")

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

        key = cv2.waitKey(1) & 0xFF

        if key == ord('v'):
            cv2.imwrite(f"{SAVE_VIOLENT}/v_{count_v}.png", motion)
            print(f"Saved VIOLENT image {count_v}")
            count_v += 1

        elif key == ord('n'):
            cv2.imwrite(f"{SAVE_NON_VIOLENT}/nv_{count_nv}.png", motion)
            print(f"Saved NON-VIOLENT image {count_nv}")
            count_nv += 1

        elif key == ord('q'):
            break

        time.sleep(0.3)

    except Exception as e:
        print("Error:", e)
        time.sleep(1)

cv2.destroyAllWindows()
