import glob
import os
import numpy as np

import cv2
import base64
from PIL import Image
import io

import facial_features

# Display the 3 color swatches (left cheek, right cheek, forehead) and the average undertone value 
# Save HTML file as "filename"
def displayUndertoneSwatched_HTML(filename, numImages):
        
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

        counter = 1
        data = []
        for image in glob.glob("./ColorCorrectedImages/*.jpg"):

            faceData = []
            f = facial_features.detect_facial_landmarks(image)   

            # leftCheek, rightCheek, forehead
            avgUndertone = 0
            for i in range(0,3):
                img = f[i+3]      
                underTone = facial_features.under_tone(img)
                faceData.append((img, underTone))   
                avgUndertone +=  underTone

            avgUndertone = avgUndertone/3
            faceData.append(("", avgUndertone))
            data.append(faceData)

            print(counter)

            if counter == numImages:
                break
            counter+=1
            

        

        sorted_arr = sorted(data, key=lambda x: x[3][1], reverse=True)
        for tup in sorted_arr:
            
            # leftCheek, rightCheek, forehead
            for i in range(0,3):
                img = tup[i][0]
                _, buffer = cv2.imencode('.jpg', img)
                img_str = base64.b64encode(buffer).decode("utf-8")
                rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')

            aValue = tup[3][1]
            rows.append(f'<td>a = {aValue}</td>')
            rows.append('</tr><tr>')
            

        html = html_template.format(rows="\n".join(rows))
        # Write the HTML to a file
        filename = filename + ".html"
        with open(filename, "w") as f:
            f.write(html)

#displayUndertoneSwatched_HTML("test", 5)




# Display the 3 color swatches (left cheek, right cheek, forehead) and the average undertone value 
# Save HTML file as "filename"
def displayEyeInfo_HTML(filename, numImages):
        
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

        counter = 1
        data = []
        for image in glob.glob("./ColorCorrectedImages/*.jpg"):

            #faceData = []
            f = facial_features.detect_facial_landmarks(image)   

            eye = f[0]
            color = facial_features.find_iris(eye)
            #print(color)
            # (b, g, r), l, a, b, irisMask
            faceData = [color[0], color[1], color[2], color[3], color[4], eye]
            data.append(faceData)

            print(counter)
            if counter == numImages:
                break
            counter+=1
                    

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
        
#displayEyeInfo_HTML("test", 10)



def displayHairInfo_HTML(filename, numImages):
        
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

        counter = 1
        data = []
        for image in glob.glob("./ColorCorrectedImages/*.jpg"):

            # Load the original image
            original_image = cv2.imread(image)

            color = facial_features.getHair(image)
            # (b, g, r), l, a, b, irisMask
            rgb_color = (int(color[0][0]), int(color[0][1]), int(color[0][2]))
            img = Image.new('RGB', (50, 50), rgb_color)
            img_np = np.array(img)
            hairData = [color[0], color[1], color[2], color[3], img_np, original_image, color[4]]
            

           
            data.append(hairData)

            if counter == numImages:
                break
            counter+=1
            

        # W.R.T s
        
        sorted_arr = sorted(data, key=lambda x: x[2], reverse=True)
        
        for ar in sorted_arr:
            # ORIGINAL IMAGE
            img = ar[5]
            _, buffer = cv2.imencode('.jpg', img)
            img_str = base64.b64encode(buffer).decode("utf-8")
            rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}" width="{original_image.shape[1]//8}" height="{original_image.shape[0]//8}"></td>')

            img = ar[6]
            _, buffer = cv2.imencode('.jpg', img)
            img_str = base64.b64encode(buffer).decode("utf-8")
            rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}" width="{img.shape[1]//2}" height="{img.shape[0]//2}"></td>')

            img = ar[4]
            _, buffer = cv2.imencode('.jpg', img)
            img_str = base64.b64encode(buffer).decode("utf-8")
            rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}" width="{2*img.shape[1]}" height="{2*img.shape[0]}"></td>')


            rows.append(f'<td>(L,a,b) = ({ar[1]:.2f}, {ar[2]}, {ar[3]})</td>')


            bgrVal = ar[0]
            rows.append(f'<td>(R,G,B) = ({bgrVal[2]},{bgrVal[1]},{bgrVal[0]})</td>')
            rows.append('</tr><tr>')
        

        html = html_template.format(rows="\n".join(rows))
        # Write the HTML to a file
        filename = filename + ".html"
        with open(filename, "w") as f:
            f.write(html)

#displayHairInfo_HTML("test", 100)

def displayAllFeatures_HTML(filename, startImg, EndImg):
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

    counter = 1
    #data = []
    #for image in glob.glob("./ColorCorrectedImages/*.jpg"):
    for i in range(startImg, EndImg):
        image_str = "ColorCorrectedImages/CCF" + str(i) + ".jpg"

        
        image = image_str
        # Load the original image
        original_image = cv2.imread(image)

        #faceData = []
        f = facial_features.detect_facial_landmarks(image)   

        eyeLeft = f[0]
        eyeRight = f[1]
        cheekLeft = f[3]
        cheekRight = f[4]
        forehead = f[5]

        # ORIGINAL IMAGE
        _, buffer = cv2.imencode('.jpg', original_image)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<div>Original Image</div>')
        rows.append(f'<div><img src="data:image/jpeg;base64,{img_str}" width="{original_image.shape[1]//4}" height="{original_image.shape[0]//4}"></div>')
        rows.append('</tr><tr>')

        # SKIN (FOREHEAD, LEFT CHEEK, RIGHT CHEEK)
        # leftCheek, rightCheek, forehead
        avgUndertone = 0
        for i in range(0,3):
            img = f[i+3]      
            underTone = facial_features.under_tone(img)
            avgUndertone +=  underTone
        avgUndertone = avgUndertone/3

        rows.append('<table>')
        _, buffer = cv2.imencode('.jpg', forehead)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<td>Forehead -- Left cheek -- Right cheek</td>')
        rows.append('</tr><tr>')
        rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')
        _, buffer = cv2.imencode('.jpg', cheekLeft)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')
        _, buffer = cv2.imencode('.jpg', cheekRight)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')
        rows.append(f'<td>a = {avgUndertone:.2f}</td>')
        rows.append('</tr><tr>')
        rows.append('</table>')




        # EYES 
        color = facial_features.find_iris(eyeLeft)
        faceData = [color[0], color[1], color[2], color[3], color[4]]

        rows.append('<table>')
        _, buffer = cv2.imencode('.jpg', eyeLeft)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<td>Left eye -- Right eye</td>')
        rows.append('</tr><tr>')
        rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')
        _, buffer = cv2.imencode('.jpg', eyeRight)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')
        
        img = faceData[4]
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}" width="{4*img.shape[1]}" height="{4*img.shape[0]}"></td>')

        rows.append(f'<td>(L,a,b) = ({faceData[1]:.2f}, {faceData[2]:.2f}, {faceData[3]:.2f})</td>')
        bgrVal = faceData[0]
        rows.append(f'<td>(R,G,B) = ({bgrVal[2]},{bgrVal[1]},{bgrVal[0]})</td>')

        rows.append('</tr><tr>')
        rows.append('</table>')


        # HAIR
        color = facial_features.getHair(image)
        rgb_color = (int(color[0][0]), int(color[0][1]), int(color[0][2]))
        img = Image.new('RGB', (50, 50), rgb_color)
        img_np = np.array(img)
        # (b, g, r), l, a, b, irisMask
        #hairData = [color[0], color[1], color[2], color[3], img_np, color[4]]
        rows.append('<table>')

        _, buffer = cv2.imencode('.jpg', color[4])
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<td>Hair</td>')
        rows.append('</tr><tr>')
        rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')

        _, buffer = cv2.imencode('.jpg', img_np)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')
        rows.append(f'<td>(L,a,b) = ({color[1]:.2f}, {color[2]}, {color[3]})</td>')


        rows.append('</tr><tr>')
        rows.append('</table>')

        

        """
        if counter == numImages:
            break
        counter+=1
        """


    html = html_template.format(rows="\n".join(rows))
    # Write the HTML to a file
    filename = filename + ".html"
    with open(filename, "w") as f:
        f.write(html)
     
#imgage_str = "ColorCorrectedImages/CCF35.jpg"
displayAllFeatures_HTML("test", 50, 51)


# counter = 0
# for image in glob.glob("./ColorCorrectedImages/*.jpg"):

#     color = facial_features.getHair(image)
    
#     print(color)

#     counter+=1
#     if counter == 1:
#         break