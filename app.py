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

app = Flask(__name__, static_folder="static")
load_dotenv()

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

mail = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
csrf = CSRFProtect(app)
app.permanent_session_lifetime = timedelta(minutes=30)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@15.164.4.130:3306/desk'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

def send_to_runpod(image_path, handedness, lifestyle, purpose):
    runpod_url = "https://zyek3om6cpaa60-80.proxy.runpod.net/predict"
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
            print("[EC2] RunPod 응답 결과:", result)
            return result
        except Exception as e:
            print("❌ RunPod 요청 실패:", str(e))
            return {
                "score": 0,
                "feedback": ["RunPod 요청 실패: " + str(e)],
                "breakdown": "error",
                "image_path": ""
            }

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

    print("[EC2] RunPod에 분석 요청 전송 중...")
    result = send_to_runpod(
        image_path=upload_path,
        handedness=hand,
        lifestyle=lifestyle,
        purpose=','.join(purpose_list)
    )

    print("[EC2] 분석 결과:", result)
    score = result.get("score", 0)
    feedback = result.get("feedback", ["피드백 없음"])
    image_path = result.get("image_path", "")
    image_base64 = result.get("image_base64", "")

    image_filename = result.get("image_filename", uuid.uuid4().hex + ".jpg")
    if image_base64:
        decoded_image = base64.b64decode(image_base64)
        ec2_image_path = os.path.join("static/uploads", image_filename)
        with open(ec2_image_path, "wb") as f:
            f.write(decoded_image)
        result["image_path"] = ec2_image_path
        image_path = ec2_image_path

    new_image = Image(
        이미지ID=uuid.uuid4().hex,
        사용자ID=user_id,
        이미지경로=image_path,
        업로드일시=datetime.datetime.now()
    )
    db.session.add(new_image)
    db.session.commit()
    image_id = new_image.이미지ID

    new_rec = Recommendation(
        추천ID=uuid.uuid4().hex,
        사용자ID=user_id,
        이미지ID=image_id,
        정돈점수=score,
        피드백='\n'.join(feedback),
        추천일시=datetime.datetime.now()
    )
    db.session.add(new_rec)
    db.session.commit()

    print("✅ DB 저장 완료")
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
