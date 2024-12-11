# predict.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from image_processing import load_data

def predict_face_shape(image_path, model_path="efficientnet_face_shape_model.keras"):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB 변환 추가
    img = cv2.resize(img, (224, 224))
    img = np.array(img) / 255.0  # 정규화
    img = np.expand_dims(img, axis=0)  # 배치 차원 추가

    model = load_model(model_path)
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction, axis=1)[0]

    _, label_encoder = load_data(r"C:\Users\0323c\hair_style_recommendation\termproject\FaceShape_Dataset\training_set")
    predicted_face_shape = label_encoder.inverse_transform([predicted_label])

    return predicted_face_shape[0]