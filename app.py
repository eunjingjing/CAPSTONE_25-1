from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# 세션 관리
app.secret_key = 'aebeole_secret_key'

# DB 연결
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@15.164.4.130:3306/desk'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

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

# 비밀번호 찾기 페이지
@app.route('/find-password')
def find_password():
    return render_template('find_password.html')

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

# 회원가입 라우터
@app.route('/sign-in', methods=['GET', 'POST'])
def sign_in():
    if request.method == 'POST':
        user_id = request.form['username']
        password = request.form['password']
        email = request.form['email']
        name = request.form['name']
        
        # 중복 아이디 확인
        existing_user = User.query.filter_by(사용자ID=user_id).first()
        if existing_user:
            return "이미 사용 중인 아이디입니다.", 409  # HTTP 409

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

        return '', 200  # 정상 회원가입 시 200 응답
    return render_template('sign_in.html')

# 로그인 라우터
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['username']
        password = request.form['password']

        # DB에서 사용자 검색
        user = User.query.filter_by(사용자ID=user_id).first()

        if user and check_password_hash(user.비밀번호해시, password):
            # 로그인 성공 -> 세션 저장
            session['user_id'] = user.사용자ID
            session['user_name'] = user.이름
            return redirect(url_for('index'))
        else:
            return "로그인 실패: 아이디 또는 비밀번호 확인", 401

    return render_template('login.html')

# 마이페이지 라우터 수정
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
            'image_path': url_for('static', filename=row.이미지경로),
            'upload_date': row.추천일시.strftime('%Y-%m-%d %H:%M:%S'),
            'score': row.정돈점수,
            'comment': row.피드백 if row.피드백 else '-'
        })

    return render_template('my_page.html', records=record_list)

# # 마이 페이지 라우터
# @app.route('/my_page')
# def my_page():
#     if 'user_id' not in session:
#         return redirect(url_for('login'))
    
#     user_id = session['user_id']

#     # # 해당 사용자에 대한 추천이력 조회
#     # records = Recommendation.query.filter_by(사용자ID=user_id).order_by(Recommendation.추천일시.desc()).all()

#     # 추천이력과 이미지 테이블을 조인하여 조회
#     sql = text("""
#         SELECT 
#             추천이력.추천ID, 추천이력.이미지ID, 추천이력.정돈점수, 추천이력.피드백, 추천이력.추천일시,
#             이미지.이미지경로
#         FROM 추천이력
#         JOIN 이미지 ON 추천이력.이미지ID = 이미지.이미지ID
#         WHERE 추천이력.사용자ID = :user_id
#         ORDER BY 추천이력.추천일시 DESC
#     """)

#     result = db.session.execute(sql, {'user_id': user_id})
#     records = result.fetchall()

#     # 템플릿으로 넘길 dict 가공
#     record_list = [
#         {
#             'image_path' : r.이미지경로,
#             'upload_date': r.추천일시.strftime('%Y-%m-%d %H:%M:%S'),
#             'score': r.정돈점수,
#             'comment': r.피드백
#         } 
#         for r in records
#     ]

#     return render_template('my_page.html', records=record_list)

# # ✅ 이미지 경로 변환 (static 경로로 변환용)
# def convert_image_path(raw_path):
#     """
#     DB에는 /home/ec2-user/data/images/1.jpeg 이런 식으로 저장돼있으므로
#     웹에선 static 접근이 되도록 변환
#     """
#     filename = raw_path.split('/')[-1]
#     return url_for('static', filename='uploads/' + filename)

# # 마이 페이지 라우터 (수정 버전)
# @app.route('/my_page')
# def my_page():
#     if 'user_id' not in session:
#         return redirect(url_for('login'))
    
#     user_id = session['user_id']

#     # 추천이력과 이미지 테이블을 조인하여 조회
#     sql = text("""
#         SELECT 
#             추천이력.추천ID, 추천이력.이미지ID, 추천이력.정돈점수, 추천이력.피드백, 추천이력.추천일시,
#             이미지.이미지경로
#         FROM 추천이력
#         JOIN 이미지 ON 추천이력.이미지ID = 이미지.이미지ID
#         WHERE 추천이력.사용자ID = :user_id
#         ORDER BY 추천이력.추천일시 DESC
#     """)

#     result = db.session.execute(sql, {'user_id': user_id})
#     records = result.fetchall()

#     # DB 결과 가공
#     record_list = []
#     for row in records:
#         record_list.append({
#             'image_path': convert_image_path(row['이미지경로']),
#             'upload_date': row['추천일시'].strftime('%Y-%m-%d %H:%M:%S'),
#             'score': row['정돈점수'],
#             'comment': row['피드백'] if row['피드백'] else '-'
#         })

#     return render_template('my_page.html', records=record_list)

# 로그아웃 라우터
@app.route('/logout')
def logout():
    session.clear()  # 세션 초기화
    return redirect(url_for('index'))

# run
if __name__ == '__main__':
    app.run(port=5000, debug=True)
