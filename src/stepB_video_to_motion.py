import cv2
import os
import numpy as np


INPUT_DIR = r"D:\esp32_cam_project\Real Life Violence Dataset"
OUTPUT_DIR = r"D:\esp32_cam_project\processed_dataset"

IMG_SIZE = 96
FRAME_SKIP = 5   # take one motion sample every 5 frames

os.makedirs(f"{OUTPUT_DIR}/Violence", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/Non_Violence", exist_ok=True)

def process_videos(label):
    input_path = os.path.join(INPUT_DIR, label)
    output_path = os.path.join(OUTPUT_DIR, label)

    count = 0

    for video_file in os.listdir(input_path):
        video_path = os.path.join(input_path, video_file)
        cap = cv2.VideoCapture(video_path)

        prev_gray = None
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

            if prev_gray is None:
                prev_gray = gray
                continue

            motion = cv2.absdiff(gray, prev_gray)
            prev_gray = gray

            save_path = os.path.join(output_path, f"{label}_{count}.png")
            cv2.imwrite(save_path, motion)
            count += 1

        cap.release()

    print(f"{label}: Saved {count} motion images")

process_videos("Violence")
process_videos("Non_Violence")
