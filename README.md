# 얼굴형 분석 및 남성 헤어스타일 추천 웹 애플리케이션 💇‍♂️

이 프로젝트는 사용자가 업로드한 얼굴 사진을 분석하여 얼굴형을 분류하고, 해당 얼굴형에 적합한 남성 헤어스타일을 추천하는 웹 애플리케이션입니다. TensorFlow와 Flask를 활용하여 딥러닝 모델을 학습시키고, 이를 웹 서비스로 제공합니다.

---

## 주요 기능 ✨

1. **얼굴형 분류 모델 학습**:
   - EfficientNetB0을 사용한 얼굴형 분류 모델을 학습합니다.
   - 5가지 얼굴형(Oval, Square, Round, Heart, Triangle)을 분류합니다.

2. **사진 업로드 및 얼굴형 분석**:
   - 사용자가 업로드한 얼굴 사진에서 얼굴형을 분석합니다.
   - 분석된 결과를 기반으로 적합한 남성 헤어스타일을 추천합니다.

3. **웹 애플리케이션 제공**:
   - Flask 기반 웹 서버에서 사용자가 업로드한 이미지를 처리합니다.
   - HTML 템플릿을 사용하여 결과를 시각적으로 제공합니다.

---

## 파일 구조 📁

```
project/
├── app.py                # Flask 서버 및 라우팅
├── image_processing.py   # 데이터 로드 및 전처리
├── model_training.py     # 모델 학습 및 저장
├── predict.py            # 얼굴형 예측 및 라벨 인코더 관리
├── templates/
│   ├── index.html        # 메인 페이지
│   └── result.html       # 결과 페이지
├── static/
│   └── uploads/          # 업로드된 이미지 저장 폴더
└── efficientnet_face_shape_model_updated.keras  # 학습된 모델 파일
```


---

## 설치 및 실행 방법 🚀

### 1. 요구 사항

다음 Python 라이브러리를 설치해야 합니다:

- Python 3.8 이상
- TensorFlow
- Flask
- OpenCV
- NumPy
- Scikit-learn

```bash
pip install tensorflow flask opencv-python-headless numpy scikit-learn
```

### 2. 데이터셋 준비 📊

- Kaggle 등에서 제공되는 얼굴형 분류 데이터셋을 준비합니다.
- 데이터셋의 구조는 다음과 같아야 합니다:
  ```
  dataset/
  ├── Oval/
  ├── Square/
  ├── Round/
  ├── Heart/
  └── Triangle/
  ```

- 데이터셋 경로를 `image_processing.py`와 `model_training.py`에 설정합니다.

### 3. 모델 학습 🧠

1. `model_training.py`를 실행하여 모델을 학습시킵니다:
   ```bash
   python model_training.py
   ```
2. 학습된 모델이 `efficientnet_face_shape_model_updated.keras`로 저장됩니다.

### 4. 서버 실행 🌐

1. `app.py`를 실행하여 Flask 서버를 시작합니다:
   ```bash
   python app.py
   ```
2. 브라우저에서 `http://127.0.0.1:5000/`로 이동합니다.

### 5. 얼굴형 분석 및 추천 📸

1. 메인 페이지에서 얼굴 사진을 업로드합니다.
2. 결과 페이지에서 예측된 얼굴형과 추천 헤어스타일을 확인합니다.

---

## 코드 설명 💻

### 1. `image_processing.py`
- 얼굴 사진을 로드하고, 크기를 조정하며 정규화합니다.
- 라벨 인코딩을 사용하여 얼굴형을 숫자로 변환합니다.

### 2. `model_training.py`
- EfficientNetB0 기반 모델을 정의하고 학습시킵니다.
- 데이터 증강(Image Augmentation)을 통해 모델의 일반화 성능을 높입니다.

### 3. `predict.py`
- 학습된 모델을 사용하여 얼굴형을 예측합니다.
- 얼굴을 검출하고 얼굴 부분만 크롭한 후 모델에 입력합니다.

### 4. `app.py`
- Flask 서버로 사용자 인터페이스를 제공합니다.
- 업로드된 파일을 저장하고, 얼굴형 분석 결과를 반환합니다.

### 5. 템플릿 파일
- `index.html`: 파일 업로드 폼을 제공합니다.
- `result.html`: 얼굴형 분석 결과와 추천 헤어스타일을 표시합니다.

---

## 참고 📚

- 얼굴 검출에는 OpenCV의 Haar Cascade를 사용합니다.
- 모델 학습에는 전이 학습(Transfer Learning) 기법이 사용되었습니다.
- 추천 헤어스타일은 각 얼굴형의 특징에 기반한 일반적인 권장 사항입니다.

---

## 프로젝트 한계 및 개선 사항 🛠️

1. **한정된 헤어스타일 추천**:
   - 현재는 남성 헤어스타일만 추천하고 있습니다.
   - 여성 헤어스타일은 종류가 매우 다양하여 데이터 수집 및 추천 로직 구현이 어렵습니다.

2. **사진 품질 의존성**:
   - 낮은 해상도의 사진이나 얼굴이 잘 보이지 않는 경우 분석 정확도가 낮아질 수 있습니다.

3. **성별 구분 없는 얼굴형 분석**:
   - 얼굴형 분류는 성별, 나이에 상관없이 모두 적용 가능합니다.
   - 그러나 추천 헤어스타일은 남성을 대상으로 설계되었습니다.

---

## 개발 환경 🛠️

- Python 3.8
- Visual Studio Code
- Windows 10

---

## 기여 방법 🤝

이 프로젝트에 기여하려면:

1. 이 저장소를 포크합니다.
2. 새로운 브랜치를 만듭니다 (`git checkout -b feature/새로운기능`).
3. 변경 사항을 커밋합니다 (`git commit -am 'Add 새로운기능'`).
4. 브랜치에 푸시합니다 (`git push origin feature/새로운기능`).
5. Pull Request를 생성합니다.

---

## 라이선스 📄

이 프로젝트는 MIT 라이선스에 따라 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

