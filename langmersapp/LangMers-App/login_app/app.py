from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    if username == 'admin' and password == 'password':
        return 'Login successful!'
    else:
        return 'Invalid username or password.'

if __name__ == '__main__':
    app.run(host='5.0.0.0')
