from flask import Flask, request, render_template, jsonify
from predict import predict_face_shape
import os

app = Flask(__name__)

# 업로드된 파일을 저장할 폴더
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 이미지 확장자 제한 (jpg, jpeg, png)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# 확장자 확인 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '파일을 선택해주세요.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': '파일을 선택해주세요.'}), 400

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # 얼굴형 예측
        try:
            face_shape = predict_face_shape(filename)
            hairstyle_recommendation = get_hairstyle_recommendation(face_shape)
            return render_template('result.html', face_shape=face_shape, hairstyle=hairstyle_recommendation, image_file=file.filename)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': '유효하지 않은 파일입니다. JPG, JPEG, PNG 파일만 업로드 가능합니다.'}), 400

# 얼굴형에 맞는 헤어스타일 추천
def get_hairstyle_recommendation(face_shape):
    recommendations = {
        'Oval': '타원형 얼굴형은 균형 잡힌 형태로, 긴 머리나 숏컷 모두 잘 어울립니다. 다양한 스타일링이 가능합니다!',
        'Square': '사각형 얼굴형은 스퀘어 컷, 콤마 스타일, 사이드 프론트 스타일이 어울립니다.',
        'Round': '둥근 얼굴형은 긴 머리 스타일과 사이드 파트가 잘 어울립니다.',
        'Heart': '하트형 얼굴형은 앞머리를 내리는 스타일이 잘 어울립니다.',
        'Triangle': '삼각형 얼굴형은 사이드 파트와 양옆이 짧은 머리 스타일이 어울립니다.'
    }
    return recommendations.get(face_shape, '헤어스타일 추천을 찾을 수 없습니다.')

if __name__ == '__main__':
    app.run(debug=True)
