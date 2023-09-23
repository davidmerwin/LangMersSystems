from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

users = {
    "admin": generate_password_hash("password123")
}

def authenticate(username, password):
    if username in users and check_password_hash(users[username], password):
        return True
    return False

def register(username, password):
    if username in users:
        return False

    users[username] = generate_password_hash(password)
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if authenticate(username, password):
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Login failed. Invalid username or password.', 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if register(username, password):
            flash('Registration successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Registration failed. The username is already taken.', 'danger')

    return render_template('register.html')

if __name__ == '__main__':
    app.run(debug=True)
