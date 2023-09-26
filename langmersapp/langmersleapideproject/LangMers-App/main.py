from flask import Flask, render_template, request
app = Flask(__name__)
@app.route('/')
def home():
    return 'Welcome to my login app!'
@app.route('/login')
def login():
    return render_template('login.html')
