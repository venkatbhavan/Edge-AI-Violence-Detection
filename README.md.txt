# 🛡️ Edge-AI Based Real-Time Violence Detection using ESP32-CAM

**Author:** Venkat Bhavan Tati  
**Matrikel-Nr:** 1575011  
**Course:** Industrial AI – Edge AI in Industrial Applications (AI5252)  
**Semester:** WiSe 25/26  

---

## 📌 Overview

This project implements a distributed **Edge-AI system** for real-time violence detection using:

- 📷 ESP32-CAM (Edge Sensing Device)
- 🧠 MobileNetV2-based CNN (Edge Inference Model)
- 🌐 Embedded Web Interface (Flask Server)
- 📱 Multi-device remote access (Mobile / Tablet)

The system performs real-time classification of violent vs non-violent actions using motion-based preprocessing and lightweight deep learning inference.

The architecture demonstrates a **multi-edge deployment strategy** suitable for industrial safety monitoring environments.

---

# 🏗️ System Architecture

The system consists of two edge devices and remote client devices:

### 🔹 Edge Device 1 – ESP32-CAM
- Captures live video
- Streams frames via Wi-Fi
- Performs no AI inference

### 🔹 Edge Device 2 – Edge Inference Node (Laptop)
- Motion preprocessing
- CNN inference (MobileNetV2)
- Threshold-based decision logic
- Embedded web service hosting

### 🔹 Remote Client Devices
- Smartphone / Tablet
- Access live results via IP-based web interface
- No inference performed on client

---

## 🔄 System Workflow

ESP32-CAM → Wi-Fi → Edge Inference Node → CNN Classification → Web Interface → Mobile Client


1. ESP32-CAM captures frames  
2. Frames transmitted via HTTP  
3. Motion preprocessing applied  
4. CNN performs inference  
5. Violence probability computed  
6. Results visualized locally  
7. Embedded server streams output  
8. Mobile device accesses via browser  

---

# 🧠 AI Methodology

## Binary Classification

- Class 0 → Non-Violence  
- Class 1 → Violence  

---

## Motion-Based Representation

To emphasize dynamic behavior:

Motion(t) = |Frame(t) − Frame(t−1)|


This:
- Suppresses static background
- Reduces lighting sensitivity
- Highlights aggressive movement patterns

---

## Model Architecture

- Base Model: **MobileNetV2**
- Transfer Learning from ImageNet
- Input: 96 × 96 grayscale
- Output: Sigmoid activation
- Lightweight architecture suitable for edge deployment

---

# 📊 Model Performance

Total samples: **55,085**

| Metric | Value |
|--------|-------|
| Accuracy | 96% |
| Precision (Violence) | 0.95 |
| Recall (Violence) | 0.97 |
| Weighted F1-Score | 0.9565 |

Low false negative rate is especially important for safety-critical monitoring systems.

---

# 🌐 Embedded Edge Deployment

The trained model is deployed using a lightweight Flask server.

### Run the Web Server

```bash
python stepG_web_embedded_violence.py
Then open in your browser:

http://<your-ip-address>:5000
Accessible from any device on the same local network.

This ensures:

No cloud dependency

Local inference only

Multi-device accessibility

🔐 GDPR & Privacy Considerations
The system follows privacy-by-design principles:

No persistent storage of video data

Frames processed in volatile memory only

No cloud transmission

No facial recognition

No biometric identification

Operates within local network

The system performs scene-level classification only and is intended as an assistive monitoring tool.

📁 Repository Structure
esp32_cam_project/
│
├── model/
│   └── action_violence_model.h5
│
├── src/
│   ├── step3_capture_frames.py
│   ├── step4_motion_preprocessing.py
│   ├── step5_collect_action_dataset.py
│   ├── stepC_train_action_model.py
│   ├── stepD_live_violence_detection.py
│   ├── stepF_esp32_cam_violence_detection.py
│   ├── stepG_web_embedded_violence.py
│   └── stepI_app_phone_camera_violence.py
│
├── results/
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
├── README.md
└── requirements.txt
📝 Dataset
The dataset is not included in this repository due to size and licensing constraints.

To reproduce training:

Download a public violence dataset.

Convert videos to frames.

Apply motion preprocessing.

Place processed data in:

processed_dataset/
    ├── violence/
    └── non_violence/
⚙️ Installation
1️⃣ Create Environment
conda create -n violence_ai python=3.9
conda activate violence_ai
2️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Running the System
Laptop Camera
python stepD_live_violence_detection.py
ESP32-CAM
python stepF_esp32_cam_violence_detection.py
Embedded Web Server
python stepG_web_embedded_violence.py
🎯 Conclusion
This project demonstrates a distributed Edge-AI system combining:

Edge sensing (ESP32-CAM)

Lightweight CNN inference

Real-time classification

Embedded web deployment

Multi-device accessibility

GDPR-conscious design

The architecture reflects practical industrial Edge-AI deployment strategies and provides a scalable foundation for further enhancements.