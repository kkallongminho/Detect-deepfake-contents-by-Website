# 🧠 Deepfake Detection System (Image + Audio) - Django Backend

This is a multi-modal deepfake detection project built using Django. It detects fake media by analyzing both facial images and voice audio. Based on uploaded `.jpg`, `.png`, or `.mp4` files, the system performs face detection, frame extraction, MFCC feature analysis, and classifies the input as **Real** or **Fake** using EfficientNet-B3 (PyTorch) and LSTM (TensorFlow) models.

---

## ⚙️ Features

| Module       | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| 🎞 Image      | Face detection (OpenCV) → crop → EfficientNet-B3 inference (PyTorch)        |
| 🔊 Audio      | Extract audio from .mp4 → MFCC → LSTM inference (TensorFlow `.h5` model)     |
| 🖼 Video       | Extract 10 representative frames → crop faces → run image prediction        |
| 💾 Storage     | Automatically saves processed media and results to `/media/processed/`      |
| 🖥 UI Support  | Upload and result pages via `upload.html` and `result.html`                 |

---

## 📦 Setup Requirements

```bash
pip install torch torchvision tensorflow opencv-python pillow librosa pydub django
```

---

## 🧠 Model Overview

| Model         | Framework | Input Format         | Output         |
|---------------|-----------|----------------------|----------------|
| EfficientNetB3| PyTorch   | (300, 300, 3) image  | Sigmoid(0~1)   |
| LSTM + MFCC   | TensorFlow| MFCC (40, time, 1)   | Sigmoid(0~1)   |

Trained weights should be saved under `/models/`:
- `best_EfficientNetB3_2data_CEn.pth`
- `voice_detection_model.h5`

---

## 🔁 Media Processing Flow

### 🖼 Image (.jpg/.png)
1. Load EfficientNetB3 model
2. Detect face with OpenCV → crop
3. Run inference → sigmoid → Real/Fake
4. Draw bounding box with result label

### 🎥 Video (.mp4)
1. Extract audio → run LSTM model on MFCC features
2. Extract 10 frames → run face detection & EfficientNetB3
3. Average all predictions → Final video-level result

---

## 📁 Folder Structure

```
myproject/
├── detection/
│   ├── views.py             ← Main logic for detection
│   ├── templates/detection/
│   │   ├── upload.html
│   │   └── result.html
│   └── static/
├── models/
│   ├── best_EfficientNetB3_2data_CEn.pth
│   └── voice_detection_model.h5
├── media/
│   ├── audio/
│   └── processed/
└── manage.py
```

---

## 🧪 Sample Output

| Field          | Example Value                              |
|----------------|---------------------------------------------|
| image_result   | Real / Fake                                 |
| image_prob     | 0.8743                                      |
| audio_result   | Fake                                        |
| audio_prob     | 0.6132                                      |
| image_path     | /media/processed/detected_frame_1.jpg       |

---

## 🖥 User Workflow

1. User visits `/deepfake/` → Uploads a file
2. System processes the file → redirects to `/deepfake/result/`
3. Results are displayed with probabilities, preview image, and delete option

---

## 💡 Suggestions for Improvement
- Add logging and error reporting module
- Support real-time webcam stream inference
- Add REST API (`/api/detect`) support for frontend apps

---

## 📄 License & Credits
This project was developed as part of the **2025 Capstone Design** course at Hanbat National University. All data and models are based on publicly available datasets and research.

> 🏫 Capstone Design - Department of Computer Engineering, Hanbat Univ.
> all process are in 캠스톤.pptx
> .pth file is trained model for deepfake detection
> deepfake.py is for django's views.py
>
> 📁 Dataset Description

This dataset is constructed from two well-known sources: FaceForensics++ and Celeb-DF. It consists of 400 video samples in total — 200 real and 200 fake. For each video, 40 facial frames were extracted, resulting in a large collection of face images suitable for deepfake detection tasks.
	•	Each frame is preprocessed and resized to 256x256 pixels in JPG format.
	•	The “real” images correspond to authentic facial videos from the original datasets.
	•	The “fake” images were synthetically generated using face manipulation techniques in the original datasets (e.g., autoencoder-based facial swapping and GAN-based deepfake generation).

This dataset is particularly well-suited for training and evaluating binary classification models that aim to distinguish between authentic and manipulated facial images.

⸻

📌 Source Details
	•	FaceForensics++ Dataset: A benchmark dataset for facial forgery detection. It contains high-quality face manipulation videos using Deepfakes, Face2Face, FaceSwap, and NeuralTextures.
	•	Official: https://github.com/ondyari/FaceForensics
	•	Celeb-DF Dataset: A dataset with improved deepfake realism using refined synthesis methods. Celeb-DF aims to provide challenging fake videos closer to real-world conditions.
	•	Official: https://github.com/yuezunli/Celeb-DF

⸻

🔧 Preprocessing Notes
	•	All videos were face-cropped using a detection algorithm and converted to static frames.
	•	The dataset was curated and organized to maximize training utility by balancing class distribution and standardizing resolution and format.

⸻

🏷️ Labels
	•	Label: 0 → Fake (synthetically manipulated)
	•	Label: 1 → Real (unaltered, genuine videos)

