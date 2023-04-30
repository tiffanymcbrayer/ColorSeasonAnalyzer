from flask import Flask
from flask import render_template
import dataHTML
import facial_features
import base64
from decision import predict_image
from flask import Response, request, jsonify, redirect, url_for, session
import os
import cv2
from werkzeug.utils import secure_filename


app = Flask(__name__)
#data
app.secret_key = 'sk'
#ROUTES
#global var
data = []
colors = {'Winter': {
    'Pure White': '#F0EDE1',
    'Light True Gray': '#D1D3D4',
    'Medium True Gray': '#71716F',
    'Charcoal Gray': '#6C6868',
    'Black': '#111810',
    'Gray-Beige': '#927C66',
    'Navy Blue': '#403F6F',
    'True Blue': '#1F4477',
    'Ice Green': '#7ED9C3',
    'Ice Yellow': '#FEFECD',
    'Iced Aqua': '#ACD3DC',
    'Violet Ice': '#C2ACB1',
    'Icy Pink': '#EBBECC',
    'Blue Ice': '#6F7A9F',
    'Royal Blue': '#3E428B',
    'Hot Turquoise': '#26FFE9',
    'China Blue': '#546477',
    'Light True Green': '#00A35B',
    'True Green': '#089404',
    'Emerald Green': '#009874',
    'Pine Green': '#3A795E',
    'Shocking Pink': '#DE5B8C',
    'Deep Hot Pink': '#E55982',
    'Magenta': '#D0417E',
    'Fuchsia': '#DF88B7',
    'Royal Purple': '#603F83',
    'Bright Burgundy': '#64313E',
    'Blue-Red': '#47191B',
    'True Red': '#BC243C',
    'Silver-tone Accessories': '#B1B3B3'
}, 'Summer' : {
    'Soft White': '#E9E0C9',
    'Rose-Beige': '#E2CAC2',
    'Cocoa': '#6C5043',
    'Rose-Brown': '#80565B',
    'Light Blue-Gray': '#B7C9E2',
    'Charcoal Blue-Gray': '#2D4256',
    'Grayed Navy': '#656B83',
    'Gray-Blue': '#565F7E',
    'Powder Blue': '#96B3D2',
    'Sky Blue': '#8ABAD3',
    'Medium Blue': '#0000CD',
    'Periwinkle Blue': '#6667AB',
    'Pastel Aqua': '#D8F3E9',
    'Pastel Blue-Green': '#A5E3E0',
    'Middle Blue-Green': '#8DD9CC',
    'Deep Blue-Green': '#006A6E',
    'Lemon Yellow': '#FAFA33',
    'Powder Pink': '#ECB7B7',
    'Pastel Pink': '#F2B8D8',
    'Rose Pink': '#FF66CC',
    'Deep Rose': '#BB5A58',
    'Watermelon': '#FD4659',
    'Blue-Red': '#47191B',
    'Burgundy': '#64313E',
    'Lavender': '#E6E6FA',
    'Orchid Bloom': '#C5AECF',
    'Pale Mauve': '#C6A4A4',
    'Raspberry': '#D32E5E',
    'Soft Fuchsia': '#D496BD',
    'Plum': '#5A315D',
    'Silver-tone Accessories': '#B1B3B3'
}, 'Fall' : {
    'Oyster White': '#E3D9C6',
    'Warm Beige': '#D3B69C',
    'Coffee Brown': '#4A2C2A',
    'Dark Chocolate Brown': '#490206',
    'Mahogany': '#824D46',
    'Camel': '#AD8166',
    'Gold': '#C8B273',
    'Medium Warm Bronze': '#825E2F',
    'Yellow-Gold': '#E5A01D',
    'Mustard': '#D8AE48',
    'Pumpkin': '#C34121',
    'Terra Cotta': '#D38377',
    'Rust': '#B55A30',
    'Deep Peach/Apricot': '#F9C5B8',
    'Salmon': '#FAAA94',
    'Orange': '#FF5800',
    'Orange-Red': '#F05627',
    'Bittersweet Red': '#D93744',
    'Dark Tomato Red': '#CE2939',
    'Lime Green': '#32CD32',
    'Chartreuse': '#B5BF50',
    'Bright Yellow-Green': '#9ACD32',
    'Moss Green': '#857946',
    'Grayed Yellow-Green': '#A29D7D',
    'Olive Green': '#8D8B55',
    'Jade Green': '#759465',
    'Forest Green': '#228B22',
    'Turquoise': '#45B8AC',
    'Teal Blue': '#478589',
    'Deep Periwinkle Blue': '#7C83BC',
    'Gold-tone Accessories': '#B49B57'
}, 'Spring' : {
    'Ivory': '#F0DFCC',
    'Buff': '#EBC396',
    'Light Warm Beige': '#FAF0DC',
    'Camel': '#B0846A',
    'Golden Tan(honey)': '#DCCA98',
    'Medium Golden Brown': '#91672F',
    'Light Warm Gray': '#D7D2CB',
    'Light Clear Navy': '#5C6B9C',
    'Light Clear Gold': '#F1E5AC',
    'Bright Golden Yellow': '#CB8E16',
    'Pastel Yellow-Green': '#EAEDA6',
    'Medium Yellow-Green': '#9ACD32',
    'Apricot': '#FFC0A0',
    'Light Orange': '#FF7231',
    'Peach': '#F99584',
    'Clear Salmon': '#FA8072',
    'Bright Coral': '#FA7268',
    'Warm Pastel Pink': '#F6C6BD',
    'Coral Pink': '#EAAC9C',
    'Clear Bright Warm Pink': '#F6C6BD',
    'Clear Bight Red': '#F93822',
    'Orange-Red': '#F05627',
    'Medium Violet': '#4E008E',
    'Light Periwinkle Blue': '#C5CBE1',
    'Dark Periwinkle Blue': '#7C83BC',
    'Light True Blue': '#2D68C4',
    'Light Warm Aqua': '#ACD3DC',
    'Clear Bright Aqua': '#30A299',
    'Medium Warm Turquoise': '#40E0D0',
    'Gold-tone Accessories': '#B49B57'
}}


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/', methods=['POST'])
def back():
    return render_template("home.html")

@app.route('/submit', methods=["POST"])
def submit_form():
    # if request.form['action'] == 'submit' and request.form['nextpg']:
    return redirect(url_for("get_photo"))


@app.route('/upload', methods=["POST"])
def upload_photo():
    global data
    if request.method == 'POST':
        uploaded_img = request.files['upload']
        img_filename = secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join("./static/images/", img_filename))
        session["uploaded_img_file_path"] = os.path.join(
            "./static/images/", img_filename
        )

        img_path = session.get("uploaded_img_file_path", None)

        imgList = [img_filename]

        dataHTML.create_HTML_file_all_features_specific_our_photos(
            "test_server", "static/images/", imgList, True
        )
        data = facial_features.facial_features_and_values(
            f"static/images/{img_filename}", True, True, 1
        )
        _, buffer = cv2.imencode(".jpg", data["eyeLeft"])
        eyeLeft = base64.b64encode(buffer).decode("utf-8")
        _, buffer = cv2.imencode(".jpg", data["eyeRight"])
        eyeRight = base64.b64encode(buffer).decode("utf-8")
        _, buffer = cv2.imencode(".jpg", data["forehead"])
        forehead = base64.b64encode(buffer).decode("utf-8")
        _, buffer = cv2.imencode(".jpg", data["cheekRight"])
        cheekRight = base64.b64encode(buffer).decode("utf-8")
        _, buffer = cv2.imencode(".jpg", data["cheekLeft"])
        cheekLeft = base64.b64encode(buffer).decode("utf-8")
        _, buffer = cv2.imencode(".jpg", data["hairMask"])
        hairMask = base64.b64encode(buffer).decode("utf-8")

        return render_template(
            "sucessful_photo.html",
            user_image=img_path,
            forehead=forehead,
            eyeLeft=eyeLeft,
            eyeRight=eyeRight,
            cheekLeft=cheekLeft,
            cheekRight=cheekRight,
            hairMask=hairMask,
        )
    return 0


@app.route('/get_photo')
def get_photo():
    return render_template("get_photo.html")


@app.route('/color_analysis', methods=['POST'])
def color_analysis():
    global data
    global colors
    if request.method == 'POST':
        img_path = session.get('uploaded_img_file_path', None)
        
        predict = predict_image(img_path, data)
        num = int(predict)
        if num == 0:
            predict = 'Winter'
        elif num == 1:
            predict = 'Summer'
        elif num == 2:
            predict = 'Fall'
        else:
            predict = 'Spring'
            
        color_swatches = colors[predict]
        return render_template('analysis.html', predict = predict, color_swatches = color_swatches)
    return 0
        

if __name__ == '__main__':
    app.run(debug=True)
