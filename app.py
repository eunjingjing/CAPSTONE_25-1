from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

app = Flask(__name__)

# DB 연결
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:1234@localhost:3306/desk'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# 메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

# 로그인 페이지
@app.route('/login')
def login():
    return render_template('login.html')

# 회원가입 페이지
@app.route('/sign-in')
def sign_in():
    return render_template('sign_in.html')

# 비밀번호 찾기 페이지
@app.route('/find-password')
def find_password():
    return render_template('find_password.html')

# DB 연결 확인 라우트
@app.route('/testdb')
def testdb():
    try:
        db.session.execute(text('SELECT 1'))
        return 'DB 연결 성공!'
    except Exception as e:
        return f'DB 연결 실패: {e}'

if __name__ == '__main__':
    app.run(port=5000, debug=True)
