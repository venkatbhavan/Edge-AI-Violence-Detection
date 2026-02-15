import cv2
import os

BASE_PATH = r"D:\esp32_cam_project\Real Life Violence Dataset"

VIDEO_FOLDERS = {
    "Violence": "Violence_image",
    "Non_Violence": "Non_Violence_image"
}

FRAME_SKIP = 10          # take 1 frame every 10 frames
IMG_SIZE = (96, 96)      # for Edge Impulse / ESP32


def extract_rgb_frames(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for video_file in os.listdir(input_dir):
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        video_path = os.path.join(input_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        img_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # IMPORTANT: frame is COLOR (BGR)
            if frame_count % FRAME_SKIP == 0:
                frame_resized = cv2.resize(frame, IMG_SIZE)
                img_name = f"{os.path.splitext(video_file)[0]}_{img_id}.jpg"
                cv2.imwrite(os.path.join(output_dir, img_name), frame_resized)
                img_id += 1

            frame_count += 1

        cap.release()

    print(f"[DONE] Processed videos from: {input_dir}")


# Run for both classes
for video_folder, image_folder in VIDEO_FOLDERS.items():
    extract_rgb_frames(
        os.path.join(BASE_PATH, video_folder),
        os.path.join(BASE_PATH, image_folder)
    )

print("✅ All videos converted to RGB images successfully")
