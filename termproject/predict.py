import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # 수정된 부분
from image_processing import load_data

def predict_face_shape(image_path, model_path='efficientnet_face_shape_model_updated.keras'):
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image at {image_path} could not be loaded.")
    else:
        # 이미지 크기 조정
        img_resized = cv2.resize(img, (224, 224))
        # 정규화
        img_resized = img_resized / 255.0
        # 배치 차원 추가
        img_resized = np.expand_dims(img_resized, axis=0)
        
        # 모델 로드
        model = load_model(model_path)

        # 예측
        prediction = model.predict(img_resized)
        
        # 예측된 클래스와 그 확률 출력
        predicted_label = np.argmax(prediction, axis=1)[0]
        predicted_prob = np.max(prediction, axis=1)[0]
        print(f"Predicted label: {predicted_label}, Probability: {predicted_prob}")
        
        # 라벨 인코딩 로드
        _, label_encoder = load_data('경로를 적어주세요')  # 경로 변경 필요
        predicted_face_shape = label_encoder.inverse_transform([predicted_label])

        return predicted_face_shape[0]
