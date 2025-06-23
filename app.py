from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
from flask_wtf import CSRFProtect
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from dotenv import load_dotenv
import os
import uuid
import datetime
import requests
import base64
from werkzeug.utils import secure_filename
from recommend import recommend_for_image

# Flask 앱 생성 및 정적파일 제어
app = Flask(__name__, static_folder="static")

# .env 불러오기
load_dotenv()

# Flask 보안 키 설정 (.env에서 가져옴)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# Flask-Mail 설정
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

# 메일 및 시리얼라이저 초기화
mail = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# 디버깅용 경로 출력
print("현재 working directory:", os.getcwd())
print("Flask static folder:", app.static_folder)

# CSRF 보호
csrf = CSRFProtect(app)

# 세션 유지 시간 (30분)
app.permanent_session_lifetime = timedelta(minutes=30)

# 쿠키 보안 설정
# HTTPS 환경에서만 활성화, HTTP에서 적용하면 세션 유지 안될 수 있음
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'


# DB 설정
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@15.164.4.130:3306/desk'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# DB 초기화
db = SQLAlchemy(app)

# RunPod에 이미지 전송 함수
def send_to_runpod(image_path, handedness, lifestyle, purpose):
    runpod_url = "https://zyek3om6cpaa60-80.proxy.runpod.net/predict" # 보안 필요(.env에서 가져오기)
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {
            "handedness": handedness,
            "lifestyle": lifestyle,
            "purpose": purpose
        }
        try:
            response = requests.post(runpod_url, files=files, data=data)
            response.raise_for_status()
            result = response.json()
            required_keys = ["score", "feedback", "image_base64"]
            if not all(k in result for k in required_keys):
                raise ValueError("RunPod 응답에 필수 필드가 누락되었습니다.")
            return result
        except Exception as e:
            print("❌ RunPod 요청 실패:", str(e))
            return {
                "score": 0,
                "feedback": ["RunPod 요청 실패: " + str(e)],
                "breakdown": "error",
                "image_path": ""
            }

# DB 연결 확인 라우트
@app.route('/testdb')
def testdb():
    try:
        db.session.execute(text('SELECT 1'))
        return 'DB 연결 성공!'
    except Exception as e:
        return f'DB 연결 실패: {e}'

# 메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

# User 모델 생성
class User(db.Model):
    __tablename__ = '사용자'
    
    사용자ID = db.Column(db.String(30), primary_key=True)
    비밀번호해시 = db.Column(db.String(256), nullable=False)
    이메일 = db.Column(db.String(100), nullable=False)
    이름 = db.Column(db.String(50), nullable=False)
    주손 = db.Column(db.Enum('왼손잡이', '오른손잡이', '양손잡이'))
    사용목적 = db.Column(db.String(100))  # SET 타입은 일단 문자열로 처리 (SET은 ORM이 조금 복잡)
    라이프스타일 = db.Column(db.Enum('맥시멀리스트', '미니멀리스트'))
    자동추천여부 = db.Column(db.Boolean, default=False)

# 추천이력 모델 생성
class Recommendation(db.Model):
    __tablename__ = '추천이력'
    
    추천ID = db.Column(db.String(40), primary_key=True)
    사용자ID = db.Column(db.String(30), db.ForeignKey('사용자.사용자ID'))
    이미지ID = db.Column(db.String(40), nullable=False)
    정돈점수 = db.Column(db.Integer)
    피드백 = db.Column(db.Text)
    추천일시 = db.Column(db.TIMESTAMP)

# 이미지 모델 생성
class Image(db.Model):
    __tablename__ = '이미지'

    이미지ID = db.Column(db.String(40), primary_key=True)
    사용자ID = db.Column(db.String(30), nullable=False)
    이미지경로 = db.Column(db.Text, nullable=False)
    업로드일시 = db.Column(db.TIMESTAMP)

# 회원가입 라우터
@app.route('/sign-in', methods=['GET', 'POST'])
def sign_in():
    if request.method == 'POST':
        user_id = request.form['username']
        password = request.form['password']
        email = request.form['email']
        name = request.form['name']
        
        # 아이디 중복 확인
        if User.query.filter_by(사용자ID=user_id).first():
            return jsonify({"success": False, "message": "이미 사용 중인 아이디입니다."})
        
         # 이메일 중복 확인
        if User.query.filter_by(이메일=email).first():
            return jsonify({"success": False, "message": "이미 사용 중인 이메일입니다."})

        # 비밀번호 해싱
        hashed_pw = generate_password_hash(password)
        new_user = User(
            사용자ID=user_id,
            비밀번호해시=hashed_pw,
            이메일=email,
            이름=name
        )
        db.session.add(new_user)
        db.session.commit()

        return jsonify({"success": True, "message": "회원가입이 완료되었습니다."})
    return render_template('sign_in.html')

# 로그인 라우터
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(사용자ID=user_id).first()

        if user and check_password_hash(user.비밀번호해시, password):
            session.permanent = True
            session['user_id'] = user.사용자ID
            session['user_name'] = user.이름
            return jsonify({"success": True, "message": f"{user.이름}님 환영합니다!"})

        return jsonify({"success": False, "message": "로그인 실패: 아이디 또는 비밀번호를 확인해주세요."})

    return render_template('login.html')

# 로그아웃 라우터
@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"success": True, "message": "로그아웃 되었습니다."})

# 비밀번호 찾기 라우터
@app.route('/find-password', methods=['GET', 'POST'])
def find_password():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        name = request.form['name']

        user = User.query.filter_by(사용자ID=username, 이메일=email, 이름=name).first()

        if not user:
            return jsonify({"success": False, "message": "입력하신 정보와 일치하는 계정이 없습니다."})

        token = serializer.dumps(email, salt='reset-password')
        reset_url = url_for('reset_password', token=token, _external=True)

        msg = Message('이룸 비밀번호 재설정 링크',
                      recipients=[email],
                      body=f'비밀번호를 재설정하려면 아래 링크를 클릭하세요:\n{reset_url}\n\n이 링크는 1시간 후 만료됩니다.')
        mail.send(msg)

        return jsonify({"success": True, "message": "비밀번호 재설정 이메일을 전송했습니다."})

    return render_template('find_password.html')

# 비밀번호 재설정 라우터
@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = serializer.loads(token, salt='reset-password', max_age=3600)
    except SignatureExpired:
        return jsonify({"success": False, "message": "링크가 만료되었습니다."})
    except BadSignature:
        return jsonify({"success": False, "message": "잘못된 링크입니다."})

    if request.method == 'POST':
        new_password = request.form['password']
        confirm_password = request.form['confirm_password']

        if new_password != confirm_password:
            return jsonify({"success": False, "message": "비밀번호가 일치하지 않습니다."})

        user = User.query.filter_by(이메일=email).first()
        user.비밀번호해시 = generate_password_hash(new_password)
        db.session.commit()

        return jsonify({"success": True, "message": "비밀번호가 성공적으로 변경되었습니다."})

    return render_template('reset_password.html', token=token)

#배치 추천(RunPod 호출)
@app.route('/recommend', methods=['POST'])
def recommend():
    print("🔥 recommend() 호출됨")
    user_id = session.get('user_id', None)
    image = request.files['image']
    hand = request.form.get('hand')
    lifestyle = request.form.get('lifestyle')
    purpose_raw = request.form.get('purpose')
    purpose_list = [p.strip() for p in purpose_raw.split(',') if p.strip()]

    if not image:
        return "이미지가 업로드되지 않았습니다.", 400

    filename = uuid.uuid4().hex + os.path.splitext(image.filename)[-1]
    upload_path = os.path.join('static/uploads', filename)
    image.save(upload_path)

    print("📡 RunPod에 분석 요청 전송 중...")
    result = send_to_runpod(
        image_path=upload_path,
        handedness=hand,
        lifestyle=lifestyle,
        purpose=','.join(purpose_list)  # RunPod에서는 문자열로 받게 처리
    )
    
    # RunPod 응답 유효성 확인
    if not result.get("image_path") and not result.get("image_base64"):
        print("❌ RunPod 응답에 이미지 경로 또는 base64 인코딩 데이터가 없습니다.")
        return "이미지 분석 결과가 유효하지 않습니다.", 500

    # RunPod 응답 수신 후 → 이미지 저장
    image_filename = result.get("image_filename", uuid.uuid4().hex + ".jpg")
    image_base64 = result.get("image_base64", "")

    if image_base64:
        decoded_image = base64.b64decode(image_base64)
        ec2_image_path = os.path.join("static/uploads", image_filename)
        with open(ec2_image_path, "wb") as f:
            f.write(decoded_image)
        result["image_path"] = ec2_image_path  # HTML에서 사용할 경로로 업데이트


    print("✅ RunPod 응답 수신 완료")

    new_image = Image(
        이미지ID=uuid.uuid4().hex,
        사용자ID=user_id,
        이미지경로=result['image_path'],
        업로드일시=datetime.datetime.now()
    )
    db.session.add(new_image)
    db.session.commit()
    image_id = new_image.이미지ID

    new_rec = Recommendation(
        추천ID=uuid.uuid4().hex,
        사용자ID=user_id,
        이미지ID=image_id,
        정돈점수=result['score'],
        피드백='\n'.join(result['feedback']),
        추천일시=datetime.datetime.now()
    )
    db.session.add(new_rec)
    db.session.commit()

    print("✅ DB 저장 완료")

    # image_path 상대 경로 보정
    image_path = result['image_path']
    rel_path = image_path.split("static/")[-1] if "static/" in image_path else image_path

    return render_template('recommend_result.html',
                       result=result,
                       image_path=rel_path)

# 마이페이지 라우터
@app.route('/my_page')
def my_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']

    sql = text("""
        SELECT 
            추천이력.추천ID, 추천이력.이미지ID, 추천이력.정돈점수, 추천이력.피드백, 추천이력.추천일시,
            이미지.이미지경로
        FROM 추천이력
        JOIN 이미지 ON 추천이력.이미지ID = 이미지.이미지ID
        WHERE 추천이력.사용자ID = :user_id
        ORDER BY 추천이력.추천일시 DESC
    """)

    result = db.session.execute(sql, {'user_id': user_id})
    records = result.fetchall()

    record_list = []
    for row in records:
        record_list.append({
            'id': row.추천ID,
            'image_path': row.이미지경로,
            'upload_date': row.추천일시.strftime('%Y-%m-%d %H:%M:%S'),
            'score': row.정돈점수,
            'comment': row.피드백 if row.피드백 else '-'
        })

    return render_template('my_page.html', records=record_list)

# RunPod 외부 접근 허용
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
