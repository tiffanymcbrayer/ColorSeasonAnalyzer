from flask import Flask
from flask import render_template 
from flask import Response, request, jsonify, redirect, url_for, session
import os
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
    # if request.form['action'] == 'submit' and request.form['nextpg']:
    uploaded_img = request.files['upload']
    print('HI')
    img_filename = secure_filename(uploaded_img.filename)
    uploaded_img.save(os.path.join("./static/images/", img_filename))
    session['uploaded_img_file_path'] = os.path.join("./static/images/", img_filename)
    
    img_path = session.get('uploaded_img_file_path', None)
    
    return render_template('sucessful_photo.html', user_image = img_path)

@app.route('/get_photo')
def get_photo():
    return render_template('get_photo.html')

if __name__ == '__main__':
    app.run(debug=True)