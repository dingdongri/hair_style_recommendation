import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(dataset_path):
    images = []
    labels = []
    
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue
        
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(img_path)
                if img is not None:
                    # 이미지 크기 조정
                    img_resized = cv2.resize(img, (224, 224))
                    # 정규화
                    img_resized = img_resized / 255.0
                    images.append(img_resized)
                    labels.append(label)
    
    # 라벨 인코딩
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    encoded_labels = to_categorical(encoded_labels, num_classes=5)  # One-hot encoding
    
    return np.array(images), np.array(encoded_labels), label_encoder
