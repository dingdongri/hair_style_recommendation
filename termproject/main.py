from model_training import train_model  # 모델 훈련 함수 임포트
from tensorflow.keras.models import load_model
from predict import predict_face_shape

def main():
    # 데이터셋 경로 지정
    training_set_path = r'C:\Users\0323c\OneDrive\opensource\termproject\FaceShape_Dataset\testing_set'
    test_image_path = r'C:\Users\0323c\OneDrive\opensource\termproject\FaceShape_Dataset\testing_set'  # 예시 이미지 경로

    # 모델 훈련
    model = train_model(training_set_path)

    # 얼굴형 예측
    predicted_face_shape = predict_face_shape(test_image_path)
    print(f'Predicted face shape: {predicted_face_shape}')

if __name__ == '__main__':
    main()
