import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask import Flask
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import sqlite3

UPLOAD_FOLDER = 'static/uploads/'
LABELS=['Fire', 'Non Fire']

app = Flask(__name__)
app.secret_key = "220b266b79426de9c275562bec329cd8"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def lite_model(image):
  # Load the TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path="model/lite_fire_detection_model.tflite")
  
  interpreter.allocate_tensors()
  interpreter.set_tensor(interpreter.get_input_details()[0]['index'], image)
  interpreter.invoke()
  return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	

@app.route('/')
def home():
	return render_template('index.html')
@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("upload.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("upload.html")
    else:
        return render_template("signup.html")

@app.route('/upload_form')
def upload_form():
	return render_template('upload.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		 
		image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		data = np.asarray(image, dtype="float32" )
		resized = cv2.resize(data, (224,224), interpolation=cv2.INTER_CUBIC)
		
		#inference
		probs_lite = lite_model(resized[None, ...])[0]
		predicted_index = np.argmax(probs_lite)
		flash(LABELS[predicted_index])
		
		return render_template('upload.html', filename=filename)
		
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/index')
def index():
	return render_template('upload.html')


if __name__ == "__main__":
    app.run(debug=True)
