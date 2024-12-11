import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from image_processing import load_data

# 전역 변수로 label_encoder 선언
label_encoder = None

def load_label_encoder():
    global label_encoder
    # label_encoder가 이미 로드되었는지 확인
    if label_encoder is None:
        _, _, label_encoder = load_data(r'C:\Users\0323c\OneDrive\opensource\termproject\FaceShape_Dataset\testing_set')  # 경로 변경 필요

def predict_face_shape(image_path, model_path='efficientnet_face_shape_model_updated.keras'):
    global label_encoder  # 전역 변수로 선언된 label_encoder 사용

    # label_encoder 로드
    load_label_encoder()

    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Image at {image_path} could not be loaded.")
    
    # 정면 얼굴 검출
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # 얼굴이 1개만 있어야 한다는 조건 추가
    if len(faces) != 1:
        raise ValueError("정면 얼굴이 아니거나 얼굴이 1개 이상 있습니다.")
    
    # 얼굴만 잘라내기
    x, y, w, h = faces[0]
    img = img[y:y+h, x:x+w]
    
    # 이미지 크기 조정 및 정규화
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    
    # 모델 로드
    model = load_model(model_path)

    # 예측
    prediction = model.predict(img_resized)
    
    # 예측된 클래스와 그 확률 출력
    predicted_label = np.argmax(prediction, axis=1)[0]
    predicted_prob = np.max(prediction, axis=1)[0]
    print(f"Predicted label: {predicted_label}, Probability: {predicted_prob}")
    
    # 얼굴형 예측
    predicted_face_shape = label_encoder.inverse_transform([predicted_label])

    return predicted_face_shape[0]
