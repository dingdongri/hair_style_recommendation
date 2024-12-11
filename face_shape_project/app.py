from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from predict import predict_face_shape

app = Flask(__name__)

# 모델 파일 경로
MODEL_PATH = "face_shape_model_fold_1.keras"

# 이미지 업로드 경로 설정
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 파일 확장자 확인 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 홈 페이지 라우트
@app.route('/')
def home():
    return render_template('index.html')

# 이미지 업로드 및 예측 라우트
@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # 얼굴형 예측
        predicted_face_shape = predict_face_shape(filename, model_path=MODEL_PATH)

        return jsonify({'predicted_face_shape': predicted_face_shape})

    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    # uploads 폴더가 없으면 생성
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
