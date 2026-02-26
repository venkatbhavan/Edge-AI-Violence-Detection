import cv2
import numpy as np
import requests
import time
from tensorflow.keras.models import load_model


ESP32_CAM_URL = "http://192.168.0.137/capture"   
MODEL_PATH = "action_violence_model.h5"

IMG_SIZE = 96
THRESHOLD = 0.45           
DISPLAY_SCALE = 2.5         


print("[INFO] Loading model...")
model = load_model(MODEL_PATH)
print("[INFO] Model loaded successfully")



def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray = gray.astype("float32") / 255.0
    gray = np.expand_dims(gray, axis=-1)    
    return np.expand_dims(gray, axis=0)      


print("[INFO] Starting ESP32-CAM live detection...")

while True:
    try:
        
        response = requests.get(
            ESP32_CAM_URL,
            timeout=12
        )

        if response.status_code != 200:
            print("[WARN] Bad HTTP response")
            time.sleep(0.5)
            continue

        img_np = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if frame is None:
            print("[WARN] Frame decode failed")
            time.sleep(0.5)
            continue

        
        input_frame = preprocess(frame)
        pred = model.predict(input_frame, verbose=0)[0][0]

        if pred > THRESHOLD:
            label = f"VIOLENCE ({pred:.2f})"
            color = (0, 0, 255)
        else:
            label = f"NON-VIOLENCE ({1 - pred:.2f})"
            color = (0, 255, 0)

        
        display = cv2.resize(
            frame,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_LINEAR
        )

        cv2.putText(
            display,
            label,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3
        )

        cv2.imshow("ESP32-CAM Violence Detection", display)

        
        time.sleep(0.3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except requests.exceptions.RequestException as e:
        print("[ESP32 ERROR]", e)
        time.sleep(1)

cv2.destroyAllWindows()
