from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(port=5000, debug=True)
