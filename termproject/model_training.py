import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from image_processing import load_data
from tensorflow.keras.applications import EfficientNetB0

def train_model(dataset_path):
    # 데이터 로드
    images, encoded_labels, label_encoder = load_data(dataset_path)
    
    # 데이터 증강 설정
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 검증 데이터셋 분리
    )
    datagen.fit(images)

    # EfficientNetB0 모델 정의
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True  # 일부 층을 학습할 수 있게 설정

    # 모델 정의
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')  # 5개의 얼굴형
    ])
    
    # 모델 컴파일
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # 작은 학습률
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 모델 학습
    model.fit(
        datagen.flow(images, encoded_labels, batch_size=32, subset='training'),
        epochs=50,  # epochs 증가
        validation_data=datagen.flow(images, encoded_labels, batch_size=32, subset='validation'),  # 검증 데이터셋 추가
        validation_steps=100  # 검증 단계 설정
    )
    
    # 모델 저장
    model.save('efficientnet_face_shape_model_updated.keras')
    return model
    
