import glob
import os

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

displayUndertoneSwatched_HTML("test", 5)