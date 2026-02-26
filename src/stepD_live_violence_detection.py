import cv2
import numpy as np
import tensorflow as tf
import time

print("RUNNING WORKING BASELINE VERSION")

MODEL_PATH = "action_violence_model.h5"
IMG_SIZE = 96


model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded")


cap = cv2.VideoCapture(0)
time.sleep(1)

if not cap.isOpened():
    print("[ERROR] Cannot open laptop webcam")
    exit()

print("[INFO] Laptop webcam opened")

prev_gray = None

cv2.namedWindow("Violence Detection (Laptop Webcam)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Violence Detection (Laptop Webcam)", 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    if prev_gray is None:
        prev_gray = gray
        continue

   
    motion = cv2.absdiff(gray, prev_gray)
    prev_gray = gray

   
    motion = motion.astype("float32") / 255.0
    motion = np.expand_dims(motion, axis=(0, -1))

   
    preds = model.predict(motion, verbose=0)
    class_id = np.argmax(preds)
    confidence = preds[0][class_id]

   
    if class_id == 1:
        label = "VIOLENCE"
        color = (0, 0, 255)
    else:
        label = "NON-VIOLENCE"
        color = (0, 255, 0)

   
    cv2.putText(
        frame,
        f"{label} ({confidence:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Violence Detection (Laptop Webcam)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
