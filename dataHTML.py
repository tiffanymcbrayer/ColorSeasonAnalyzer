import glob
import os
import numpy as np

import cv2
import base64
from PIL import Image
import io
import rawpy

import facial_features
import color_corrector

# Display the 3 color swatches (left cheek, right cheek, forehead) and the average Lab values 
def display_skin_info_HTML(filename, startImg, endImg):
    # Create an HTML template for displaying the images
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>DATA</title>
    </head>
    <body>
        <table>
            {rows}
        </table>
    </body>
    </html>
    '''
    # Generate HTML code for each row of images
    rows = []

    counter = startImg
    data = []
    for i in range(startImg, endImg):
        image = "ColorCorrectedImages/CCF" + str(i) + ".jpg"
        cc_image = cv2.imread(image)


        f = facial_features.detect_facial_landmarks(cc_image)  
        imgArr = [f[3], f[4], f[5]]
        l_avg, a_avg, b_avg = facial_features.total_under_tone(imgArr)
        data.append(((l_avg, a_avg, b_avg), imgArr))
        
        print(counter)
        counter+=1
        

    
    # L - x[0][0] || a - x[0][1] || b - x[0][2]  
    sorted_arr = sorted(data, key=lambda x: x[0][2], reverse=True)
    for tup in sorted_arr:
        # leftCheek, rightCheek, forehead
        for i in range(0,3):
            img = tup[1][i]
            _, buffer = cv2.imencode('.jpg', img)
            img_str = base64.b64encode(buffer).decode("utf-8")
            rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')

        #rows.append(f'<td>a = {aValue}</td>')
        rows.append(f'<td>(L,a,b) = ({tup[0][0]:.2f}, {tup[0][1]:.2f}, {tup[0][2]:.2f})</td>')
        rows.append('</tr><tr>')
        

    html = html_template.format(rows="\n".join(rows))
    # Write the HTML to a file
    filename = filename + ".html"
    with open(filename, "w") as f:
        f.write(html)
#display_skin_info_HTML("test", 200, 205)



def display_eye_info_HTML(filename, startImg, endImg):
    # Create an HTML template for displaying the images
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>DATA</title>
    </head>
    <body>
        <table>
            {rows}
        </table>
    </body>
    </html>
    '''
    # Generate HTML code for each row of images
    rows = []

    counter = startImg
    data = []
    for i in range(startImg, endImg):
        image = "ColorCorrectedImages/CCF" + str(i) + ".jpg"
        cc_image = cv2.imread(image)


        f = facial_features.detect_facial_landmarks(cc_image)  

        eye = f[0]
        color = facial_features.find_iris(eye)
        # (b, g, r), l, a, b, irisMask
        faceData = [color[0], color[1], color[2], color[3], color[4], eye]
        data.append(faceData)

        print(counter)
        counter+=1
                
    # L - x[1] || a - x[2] || b - x[3] 
    sorted_arr = sorted(data, key=lambda x: x[1], reverse=True)
    for ar in sorted_arr:
        img = ar[5]
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')

        img = ar[4]
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}" width="{4*img.shape[1]}" height="{4*img.shape[0]}"></td>')

        rgb_color = (int(ar[0][0]), int(ar[0][1]), int(ar[0][2]))
        img = Image.new('RGB', (50, 50), rgb_color)
        img_np = np.array(img)
        _, buffer = cv2.imencode('.jpg', img_np)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')


        rows.append(f'<td>(L,a,b) = ({ar[1]:.2f}, {ar[2]}, {ar[3]})</td>')

        bgrVal = ar[0]
        rows.append(f'<td>(R,G,B) = ({bgrVal[2]},{bgrVal[1]},{bgrVal[0]})</td>')
        rows.append('</tr><tr>')
    

    html = html_template.format(rows="\n".join(rows))
    # Write the HTML to a file
    filename = filename + ".html"
    with open(filename, "w") as f:
        f.write(html)
#display_eye_info_HTML("test", 200, 205)


# Going through color correcting images and ording hair info 
def display_hair_info_HTML(filename, startImg, endImg):
    # Create an HTML template for displaying the images
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>DATA</title>
    </head>
    <body>
        <table>
            {rows}
        </table>
    </body>
    </html>
    '''
    # Generate HTML code for each row of images
    rows = []

    counter = startImg
    data = []
    for i in range(startImg, endImg):
        image = "ColorCorrectedImages/CCF" + str(i) + ".jpg"
        cc_image = cv2.imread(image)


        f = facial_features.detect_facial_landmarks(cc_image)   
        imgArr = [f[3], f[4], f[5]]
        l_avg, a_avg, b_avg = facial_features.total_under_tone(imgArr)
        
                
        threshold_value = 100
        mask = facial_features.get_hair_mask(cc_image, threshold_value)

        l_hair, a_hair, b_hair  = facial_features.getLabColorSpace(mask)
        if l_hair > 70:
                # REDO THE HAIR MASK!!
            threshold_value = 190
            mask = facial_features.get_hair_mask(image, threshold_value)
            l_hair, a_hair, b_hair  = facial_features.getLabColorSpace(mask)

        top3colors = facial_features.get_top_color(mask,num_colors=3)

        # Lab values, 
        data.append(((l_avg, a_avg, b_avg ),cc_image, mask, (l_hair, a_hair, b_hair), top3colors))

        

        print(counter)
        counter+=1

    # L - x[3][0] || a - x[3][1] || b - x[3][2]  
    sorted_arr = sorted(data, key=lambda x: x[3][0], reverse=True)
        
    for ar in sorted_arr:
        # ORIGINAL IMAGE
        img = ar[1]
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}" width="{img.shape[1]//8}" height="{img.shape[0]//8}"></td>')

        img = ar[2]
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}" width="{img.shape[1]//2}" height="{img.shape[0]//2}"></td>')

        for i in range(0,3):
            rgb_color = ar[4][i]

            img = Image.new('RGB', (50, 50), rgb_color)
            img_np = np.array(img)

            _, buffer = cv2.imencode('.jpg', img_np)
            img_str = base64.b64encode(buffer).decode("utf-8")
            rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')

        #rows.append(f'<td>(L,a,b) = ({ar[0][0]:.2f}, {ar[0][1]:.2f}, {ar[0][2]:.2f})</td>')
        rows.append(f'<td>HAIR: (L,a,b) = ({ar[3][0]:.2f}, {ar[3][1]:.2f}, {ar[3][2]:.2f})</td>')
        rows.append(f'<td> Top 3 colors: = ({ar[4]})</td>')


        # bgrVal = ar[0]
        # rows.append(f'<td>(R,G,B) = ({bgrVal[2]},{bgrVal[1]},{bgrVal[0]})</td>')
        rows.append('</tr><tr>')
    

    html = html_template.format(rows="\n".join(rows))
    # Write the HTML to a file
    filename = filename + ".html"
    with open(filename, "w") as f:
        f.write(html)

#display_hair_info_HTML("test", 200, 205)




#-----------------------------------------------------------------------------
# THESE FUNCTIONS ARE USED TO DISPLAY ALL THE FEATURES AND VALUES ON A "TEST" HTML PAGE  

def display_all_features_HTML(img_str, counter, ours, color_correct, inBGR):
    rows = []
    data = facial_features.facial_features_and_values(img_str, ours, color_correct, inBGR)

    #original_image = data['original_image']
    color_corrected_image = data['color_corrected_image']

    _, buffer = cv2.imencode('.jpg', color_corrected_image)
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(f'<div>Color Corrected Image: {counter}</div>')
    rows.append(f'<div><img src="data:image/jpeg;base64,{img_str}" width="{color_corrected_image.shape[1]//4}" height="{color_corrected_image.shape[0]//4}"></div>')
    rows.append('</tr><tr>')

    # SKIN (FOREHEAD, LEFT CHEEK, RIGHT CHEEK)
    # leftCheek, rightCheek, forehead
    rows.append('<table>')
    _, buffer = cv2.imencode('.jpg', data['forehead'])
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(f'<td>Forehead -- Left cheek -- Right cheek</td>')
    rows.append('</tr><tr>')
    rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')
    _, buffer = cv2.imencode('.jpg', data['cheekLeft'])
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')
    _, buffer = cv2.imencode('.jpg',  data['cheekRight'])
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')
    skinLab = data['skinLab']
    rows.append(f'<td>(L,a,b) = ({skinLab[0]:.2f}, {skinLab[1]:.2f}, {skinLab[2]:.2f})</td>')
    rows.append('</tr><tr>')
    rows.append('</table>')


    # EYES 

    rows.append('<table>')
    _, buffer = cv2.imencode('.jpg', data['eyeLeft'])
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(f'<td>Left eye -- Right eye</td>')
    rows.append('</tr><tr>')
    rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')
    _, buffer = cv2.imencode('.jpg', data['eyeRight'])
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')
    
    img = data['irisMask']
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}" width="{4*img.shape[1]}" height="{4*img.shape[0]}"></td>')

    eyeRGB = data['eyeRGB']
    flippedRGB = eyeRGB[::-1]
    img = Image.new('RGB', (50, 50), flippedRGB)
    img_np = np.array(img)

    _, buffer = cv2.imencode('.jpg', img_np)
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')

    eyeLab = data['eyeLab']
    rows.append(f'<td>(L,a,b) = ({eyeLab[0]:.2f}, {eyeLab[1]:.2f}, {eyeLab[2]:.2f})</td>')
    rows.append(f'<td>(R,G,B) = ({eyeRGB[0]:.2f}, {eyeRGB[1]:.2f}, {eyeRGB[2]:.2f})</td>')

    rows.append('</tr><tr>')
    rows.append('</table>')


    # HAIR
    img = data['hairMask']
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(f'<td>Hair</td>')
    rows.append('</tr><tr>')
    rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}" width="{img.shape[1]//2}" height="{img.shape[0]//2}"></td>')

    hairColors = data['hairColors']
    for i in range(0,3):
        rgb_color = hairColors[i]
        flippedRGB = rgb_color[::-1]
        img = Image.new('RGB', (50, 50), flippedRGB)
        img_np = np.array(img)

        _, buffer = cv2.imencode('.jpg', img_np)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')

    hairLab = data['hairLab']
    rows.append(f'<td>(L,a,b) = ({hairLab[0]:.2f}, {hairLab[1]:.2f}, {hairLab[2]:.2f})</td>')
    top3colors = data['hairColors']
    rows.append(f'<td> Top 3 colors: = ({top3colors})</td>')


    rows.append('</tr><tr>')
    rows.append('</table>')
    return rows


def create_HTML_file_all_features_specific_our_photos(filename, imgList, ours = True, color_correct = True, inBGR = 1):
    # Create an HTML template for displaying the images
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>DATA</title>
    </head>
    <body>
        <table>
            {rows}
        </table>
    </body>
    </html>
    '''

    # Generate HTML code for each row of images
    rows = []
    counter = 0
    # Process the files in order
    for img_str in imgList:
        img_str = "OurPhotos/" + img_str
        img_HTML_rows = display_all_features_HTML(img_str, counter, ours, color_correct, inBGR = 1)
        rows += img_HTML_rows
        print(counter)
        counter+=1

    html = html_template.format(rows="\n".join(rows))
    # Write the HTML to a file
    filename = filename + ".html"
    with open(filename, "w") as f:
        f.write(html)


def create_HTML_file_all_features_specific(filename, img_str, ours = False, color_correct = True):
    # Create an HTML template for displaying the images
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>DATA</title>
    </head>
    <body>
        <table>
            {rows}
        </table>
    </body>
    </html>
    '''

    # Generate HTML code for each row of images
    rows = []
    counter = 0
    img_HTML_rows = display_all_features_HTML(img_str, counter, ours, color_correct, inBGR = 1)
    rows += img_HTML_rows

    html = html_template.format(rows="\n".join(rows))
    # Write the HTML to a file
    filename = filename + ".html"
    with open(filename, "w") as f:
        f.write(html)


def create_HTML_file_all_features(filename, startImg, endImg, ours = False, color_correct = True):
    # Create an HTML template for displaying the images
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>DATA</title>
    </head>
    <body>
        <table>
            {rows}
        </table>
    </body>
    </html>
    '''

    # Generate HTML code for each row of images
    rows = []
    counter = startImg

    # Specify the directory path
    dir_path = "ChicagoFaceDatabaseImages"
    #dir_path = "OurPhotos"

    # Get a list of all the files in the directory
    file_list = os.listdir(dir_path)

    # Sort the list of files in alphabetical order
    sorted_files = sorted(file_list)
    sorted_files = sorted_files[startImg:endImg]


    # Process the files in order
    for file_name in sorted_files:
        img_str = "ChicagoFaceDatabaseImages/" + file_name

        img_HTML_rows = display_all_features_HTML(img_str, counter, ours, color_correct, 0)
        rows += img_HTML_rows

        print(counter)
        counter+=1

    html = html_template.format(rows="\n".join(rows))
    # Write the HTML to a file
    filename = filename + ".html"
    with open(filename, "w") as f:
        f.write(html)
     
<<<<<<< Updated upstream
=======
#imgage_str = "ColorCorrectedImages/CCF35.jpg"
displayAllFeatures_HTML("./test", 211, 221)
>>>>>>> Stashed changes



# Use this function to look through the images in the ChicagoFaceDatabaseImages start-end
# HTML filename, start img #, end img #, color_correct True or False
create_HTML_file_all_features("test", 0, 5)



# Use this function to look at one specific image
# HTML filename, color_correct True or False
# img_str = "OurPhotos/DSC06481_2.JPG"
#create_HTML_file_all_features_specific("test", img_str,  1)


# Use this function to look through all the images in OurPhotos
# imgList = ["DSC06469.JPG", "DSC06471.JPG", "DSC06473.JPG", "DSC06474.JPG","DSC06477.JPG",
#            "DSC06479.JPG", "DSC06481.JPG", "DSC06483.JPG", "DSC06484.JPG", "DSC06488.JPG",
#             "DSC06489.JPG", "DSC06491.JPG", "DSC06494.JPG"]
# imgList = ["DSC06471.JPG","DSC06473.JPG","DSC06474.JPG","DSC06477.JPG","DSC06479.JPG","DSC06481.JPG",
#            "DSC06483.JPG","DSC06484.JPG","DSC06488.JPG","DSC06489.JPG","DSC06491.JPG","DSC06494.JPG"]
# # imgList = ["DSC06473.JPG", "DSC06474.JPG", "DSC06481.JPG"]
# create_HTML_file_all_features_specific_our_photos("test_ours", imgList, True)





