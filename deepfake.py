import os
import shutil
import cv2
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
from pydub import AudioSegment
from django.conf import settings
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, HttpResponse
from torchvision import models, transforms
import librosa

# 디렉토리 설정
MEDIA_DIR = settings.MEDIA_ROOT
MEDIA_URL = settings.MEDIA_URL
AUDIO_DIR = os.path.join(MEDIA_DIR, "audio/")
IMAGE_OUTPUT_DIR = os.path.join(MEDIA_DIR, "processed/")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# 얼굴 탐지 모델 로딩
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 글로벌 모델 변수
audio_model = None
image_model = None

# 오디오 모델 로드
def load_audio_model():
    global audio_model
    if audio_model is None:
        model_path = os.path.join(settings.BASE_DIR, "models", "voice_detection_model.h5")
        audio_model = tf.keras.models.load_model(model_path, compile=False)
    return audio_model

# 이미지 모델 로드
def load_image_model():
    global image_model
    if image_model is None:
        model_path = os.path.join(settings.BASE_DIR, "models", "deepfake_efficientnetb3_finetuned.pth")
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = torch.nn.Linear(1536, 1)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")), strict=False)
        model.eval()
        image_model = model
    return image_model

# 폴더 비우기
def clear_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
    except Exception as e:
        print(f"폴더 비우기 실패: {e}")

def detect_fake_image(image_path):
    try:
        model = load_image_model()
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1️⃣ 얼굴 탐지 먼저 수행
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            print("[INFO] 얼굴을 찾지 못함, 전체 이미지 사용")
            crop_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # 여러 얼굴이 있으면 첫 번째 얼굴만 분석
            (x, y, w, h) = faces[0]
            crop_img = img[y:y+h, x:x+w]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

        # 2️⃣ 얼굴 부분을 PIL Image로 변환 후 모델 입력
        pil_image = Image.fromarray(crop_img)
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ])
        tensor_img = transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor_img)
            prediction = output.sigmoid().item()

        result = "Fake" if prediction > 0.5 else "Real"

        # 3️⃣ 결과에 따라 박스 색상 지정 후 원본 이미지에 박스 그림
        for (x, y, w, h) in faces:
            color = (0, 255, 0) if result == "Real" else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # 4️⃣ 최종 이미지 저장
        output_path = os.path.join(IMAGE_OUTPUT_DIR, "detected_" + os.path.basename(image_path))
        cv2.imwrite(output_path, img)

        return result, prediction, output_path

    except Exception as e:
        print(f"[❌ 얼굴 기반 이미지 분석 실패]: {e}")
        return "Error", 0.0, None


# 오디오 분석
def detect_fake_audio(audio_path):
    try:
        model = load_audio_model()
        audio_data = preprocess_audio(audio_path)
        prediction = model.predict(audio_data)
        result = "Fake" if prediction > 0.5 else "Real"
        return result, float(prediction)
    except Exception as e:
        print(f"오디오 분석 실패: {e}")
        return "Error", 0.0

# 오디오 전처리
def preprocess_audio(audio_path, sr=16000, n_mfcc=40, duration=3):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        max_len = sr * duration
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)))
        else:
            y = y[:max_len]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc = np.expand_dims(mfcc, axis=-1)
        return mfcc.astype(np.float32)
    except Exception as e:
        print(f"❌ MFCC 전처리 실패: {e}")
        return np.zeros((1, n_mfcc, 130, 1), dtype=np.float32)

# MP4에서 오디오 추출
def extract_audio(video_path):
    audio_path = os.path.join(AUDIO_DIR, os.path.splitext(os.path.basename(video_path))[0] + ".wav")
    try:
        audio = AudioSegment.from_file(video_path)
        if len(audio) == 0:
            return None
        audio.export(audio_path, format="wav")
        return audio_path
    except Exception:
        return None

# 비디오 프레임 추출
def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(frame_count // num_frames, 1)
    frame_paths = []
    for i in range(num_frames):
        frame_index = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(IMAGE_OUTPUT_DIR, f"frame_{i}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_paths.append(frame_filename)
    cap.release()
    return frame_paths

# 파일 업로드 처리
def handle_image(request):
    folder_path = os.path.join(settings.MEDIA_ROOT, 'processed')
    clear_folder(folder_path)

    uploaded_file = request.FILES.get("uploaded_file")
    if not uploaded_file:
        return render(request, "detection/upload.html", {"error": "❌ 업로드된 파일이 없습니다."})

    file_extension = uploaded_file.name.split(".")[-1].lower()
    allowed_extensions = ["jpg", "jpeg", "png", "mp4", "wav", "mp3"]

    if file_extension not in allowed_extensions:
        return render(request, "detection/upload.html", {
            "error": f"❌ 지원되지 않는 파일 유형: {file_extension}"
        })

    fs = FileSystemStorage(location=MEDIA_DIR)
    filename = fs.save(uploaded_file.name, uploaded_file)
    file_path = os.path.join(MEDIA_DIR, filename)

    image_result, image_prob, processed_filename = "N/A", 0.0, ""
    frame_paths = []
    audio_result, audio_prob = "N/A", 0.0

    try:
        # 이미지 파일 처리
        if file_extension in ["jpg", "jpeg", "png"]:
            image_result, image_prob, processed_filename = detect_fake_image(file_path)

        # 비디오 파일 처리
        elif file_extension == "mp4":
            audio_path = extract_audio(file_path)
            frame_paths = extract_frames(file_path, num_frames=10)
            frame_results = [detect_fake_image(frame) for frame in frame_paths if os.path.exists(frame)]
            frame_probs = [prob for _, prob, _ in frame_results if isinstance(prob, (int, float))]
            image_prob = sum(frame_probs) / len(frame_probs) if frame_probs else 0.0
            image_result = "Fake" if image_prob > 0.5 else "Real"
            for _, _, fname in frame_results:
                if fname:
                    processed_filename = fname
                    break
            if audio_path:
                audio_result, audio_prob = detect_fake_audio(audio_path)

        # 순수 오디오 파일 처리
        elif file_extension in ["wav", "mp3"]:
            audio_result, audio_prob = detect_fake_audio(file_path)

    except Exception as e:
        print(f"⚠️ 처리 중 오류 발생: {e}")

    finally:
        if processed_filename:
            relative_path = os.path.relpath(processed_filename, MEDIA_DIR)
            image_path = f"/media/{relative_path}"
        else:
            image_path = "None"

        return redirect(
            f"/deepfake/result/?image_result={image_result}&image_prob={image_prob:.4f}&image_path={image_path}"
            f"&uploaded_file={filename}&audio_result={audio_result}&audio_prob={audio_prob:.4f}"
        )

# 결과 페이지
def result_page(request):
    image_result = request.GET.get("image_result", "Unknown")
    image_prob = request.GET.get("image_prob", "N/A")
    image_filename = request.GET.get("image_path", "")
    uploaded_file = request.GET.get("uploaded_file", "")
    audio_result = request.GET.get("audio_result", "N/A")
    audio_prob = request.GET.get("audio_prob", "N/A")

    if request.method == "POST" and 'delete_file' in request.POST:
        if uploaded_file:
            file_to_delete = os.path.join(MEDIA_DIR, uploaded_file)
            try:
                if os.path.exists(file_to_delete):
                    os.remove(file_to_delete)
                    print(f"File {uploaded_file} deleted successfully.")
                else:
                    print(f"File {uploaded_file} not found.")
            except Exception as e:
                print(f"Error deleting file {uploaded_file}: {e}")
        return redirect('/')

    return render(request, "detection/result.html", {
        "image_result": image_result,
        "image_prob": image_prob,
        "image_path": image_filename,
        "uploaded_file": uploaded_file,
        "audio_result": audio_result,
        "audio_prob": audio_prob,
    })
