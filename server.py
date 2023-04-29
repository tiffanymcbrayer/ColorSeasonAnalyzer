from flask import Flask
from flask import render_template 
import dataHTML
import facial_features
import base64
import json
import numpy as np
from flask import Response, request, jsonify, redirect, url_for, session
import os
import cv2
from werkzeug.utils import secure_filename



app = Flask(__name__)
#data
app.secret_key = 'sk'
#ROUTES

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    # if request.form['action'] == 'submit' and request.form['nextpg']:
    return redirect(url_for('get_photo'))

@app.route('/upload', methods=['POST'])
def upload_photo():
    if request.method == 'POST':
        uploaded_img = request.files['upload']
        img_filename = secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join("./static/images/", img_filename))
        session['uploaded_img_file_path'] = os.path.join("./static/images/", img_filename)
        
        img_path = session.get('uploaded_img_file_path', None)
        
        imgList = [img_filename]
        
        dataHTML.create_HTML_file_all_features_specific_our_photos("test_server", "static/images/", imgList, True)
        data = facial_features.facial_features_and_values(f'static/images/{img_filename}', True, True, 1)
        _, buffer = cv2.imencode('.jpg', data['eyeLeft'])
        eyeLeft = base64.b64encode(buffer).decode("utf-8")
        _, buffer = cv2.imencode('.jpg', data['eyeRight'])
        eyeRight = base64.b64encode(buffer).decode("utf-8")
        _, buffer = cv2.imencode('.jpg', data['forehead'])
        forehead = base64.b64encode(buffer).decode("utf-8")
        _, buffer = cv2.imencode('.jpg', data['cheekRight'])
        cheekRight = base64.b64encode(buffer).decode("utf-8")
        _, buffer = cv2.imencode('.jpg', data['cheekLeft'])
        cheekLeft = base64.b64encode(buffer).decode("utf-8")
        _, buffer = cv2.imencode('.jpg', data['hairMask'])
        hairMask = base64.b64encode(buffer).decode("utf-8")
       
        return render_template('sucessful_photo.html', user_image = img_path, forehead = forehead, eyeLeft = eyeLeft, eyeRight = eyeRight, cheekLeft = cheekLeft, cheekRight = cheekRight, hairMask = hairMask)
    return 0

@app.route('/get_photo')
def get_photo():
    return render_template('get_photo.html')

if __name__ == '__main__':
    app.run(debug=True)
