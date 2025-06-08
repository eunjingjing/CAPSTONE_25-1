# 📦 CAPSTONE\_25-1 프로젝트

Flask 기반 웹 백엔드 + YOLOv8 모델 서버 연동 프로젝트

---

## 🔧 개발 환경

* Python 3.11.9
* Flask 3.1.1
* VSCode (추천)
* 가상환경 사용 (`venv`)
* YOLOv8 (Ultralytics)

---

## ⚙️ 로컬컬 개발 환경 세팅 방법 (처음 1회만)

### 1. GitHub에서 레포 복제

### 2. 가상환경 생성 및 활성화

#### 💻 Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### 🐧 macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 🚀 개발 시작

가상환경을 활성화한 후 Flask 서버를 실행하세요.

```bash
python app.py
```

또는 Flask 환경 변수 설정 후 실행:

```bash
set FLASK_APP=app.py        # Windows
export FLASK_APP=app.py     # macOS/Linux

flask run
```

---

## 📁 디렉토리 구조 예시

```
CAPSTONE_25-1/
🔹 app.py
🔹 static/
🔹 templates/
🔹 model/             # YOLOv8 모델 파일
🔹 requirements.txt
🔹 README.md
```

---

## ❗ 주의사항

* `venv/` 폴더는 `.gitignore`에 포함되어야 하며 까프핸터에 올린면 안됨.
* 팀원은 각자 로컬에서 가상환경을 생성해야 합니다.
* 모델 파일은 GitHub 용량 제한 때문에 바가이 공유해야 할 수 있습니다.

---

## 🤝 기억 방법

1. 새로운 브랜치를 만들어 개발합니다.
2. 기능 구현 후 PR(Pull Request)을 생성합니다.
3. 팀원들과 코드 리뷰 후 병합합니다.
