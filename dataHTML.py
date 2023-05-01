import os
import numpy as np
import cv2
import base64
from PIL import Image
import facial_features
import typing as npt
from typing import List, Tuple

"""
This file contains functions to create HTML files to organize and display data with the images and corresponding Lab and RGB values.
"""

def display_skin_info_HTML(filename:str, start_img:int, end_img:int) -> None:
    """
    This function generates an HTML file for the color-corrected images retrieved from the Chicago Database. 
    The file will contain the images of the person's left cheek, right cheek, and forehead, along with their corresponding Lab values. 
    The images will be arranged in ascending order based on the person's averaged L value.

    Parameters:
    ----------
    filename : str
        Name of HTML file you want to create and write to 
    start_img : int
        The index of the first image to be processed.
    end_img : int 
        The index of the last image to be processed (not including)
    """

    # Create an HTML template for displaying the images
    html_template = """
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
    """
    # Generate HTML code for each row of images
    rows = []

    counter = start_img
    data = []
    for i in range(start_img, end_img):
        image = "ColorCorrectedImages/CCF" + str(i) + ".jpg"
        cc_image = cv2.imread(image)

        f = facial_features.detect_facial_landmarks(cc_image)
        img_arr = [f[3], f[4], f[5]]
        l_avg, a_avg, b_avg = facial_features.total_under_tone(img_arr)
        data.append(((l_avg, a_avg, b_avg), img_arr))

        print(counter)
        counter += 1

    # L - x[0][0] || a - x[0][1] || b - x[0][2]
    sorted_arr = sorted(data, key=lambda x: x[0][2], reverse=True)
    for tup in sorted_arr:
        # left_cheek, right_cheek, forehead
        for i in range(0, 3):
            img = tup[1][i]
            _, buffer = cv2.imencode(".jpg", img)
            img_str = base64.b64encode(buffer).decode("utf-8")
            rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')

        # rows.append(f'<td>a = {aValue}</td>')
        rows.append(
            f"<td>(L,a,b) = ({tup[0][0]:.2f}, {tup[0][1]:.2f}, {tup[0][2]:.2f})</td>"
        )
        rows.append("</tr><tr>")

    html = html_template.format(rows="\n".join(rows))
    # Write the HTML to a file
    filename = filename + ".html"
    with open(filename, "w") as f:
        f.write(html)


def display_eye_info_HTML(filename:str, start_img:int, end_img:int) -> None:
    """
    This function generates an HTML file for the color-corrected images retrieved from the Chicago Database. 
    The file will contain the images of the left eye, iris box, example of RGB value, along with their corresponding Lab values and RGB value. 
    The images will be arranged in ascending order based on the iris box L value.

    Parameters:
    ----------
    filename : str
        Name of HTML file you want to create and write to 
    start_img : int
        The index of the first image to be processed.
    end_img : int 
        The index of the last image to be processed (not including)
    """
    # Create an HTML template for displaying the images
    html_template = """
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
    """
    # Generate HTML code for each row of images
    rows = []

    counter = start_img
    data = []
    for i in range(start_img, end_img):
        image = "ColorCorrectedImages/CCF" + str(i) + ".jpg"
        cc_image = cv2.imread(image)

        f = facial_features.detect_facial_landmarks(cc_image)

        eye = f[0]
        color = facial_features.find_iris(eye)
        # (b, g, r), l, a, b, irisMask
        face_data = [color[0], color[1], color[2], color[3], color[4], eye]
        data.append(face_data)

        print(counter)
        counter += 1

    # L - x[1] || a - x[2] || b - x[3]
    sorted_arr = sorted(data, key=lambda x: x[1], reverse=True)
    for ar in sorted_arr:
        img = ar[5]
        _, buffer = cv2.imencode(".jpg", img)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')

        img = ar[4]
        _, buffer = cv2.imencode(".jpg", img)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(
            f'<td><img src="data:image/jpeg;base64,{img_str}" width="{4*img.shape[1]}" height="{4*img.shape[0]}"></td>'
        )

        rgb_color = (int(ar[0][0]), int(ar[0][1]), int(ar[0][2]))
        img = Image.new("RGB", (50, 50), rgb_color)
        img_np = np.array(img)
        _, buffer = cv2.imencode(".jpg", img_np)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')

        rows.append(f"<td>(L,a,b) = ({ar[1]:.2f}, {ar[2]:.2f}, {ar[3]:.2f})</td>")

        bgr_val = ar[0]
        rows.append(f"<td>(R,G,B) = ({bgr_val[2]},{bgr_val[1]},{bgr_val[0]})</td>")
        rows.append("</tr><tr>")

    html = html_template.format(rows="\n".join(rows))
    # Write the HTML to a file
    filename = filename + ".html"
    with open(filename, "w") as f:
        f.write(html)


def display_hair_info_HTML(filename:str, start_img:int, end_img:int) -> None:
    """
    This function generates an HTML file for the color-corrected images retrieved from the Chicago Database. 
    The file will contain the original image, hair image, 3 examples boxes of the top 3 RGB values, along with their corresponding Lab values and top 3 RGB values. 
    The images will be arranged in ascending order based on the hair image L value.

    Parameters:
    ----------
    filename : str
        Name of HTML file you want to create and write to 
    start_img : int
        The index of the first image to be processed.
    end_img : int 
        The index of the last image to be processed (not including)
    """
    # Create an HTML template for displaying the images
    html_template = """
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
    """
    # Generate HTML code for each row of images
    rows = []

    counter = start_img
    data = []
    for i in range(start_img, end_img):
        image = "ColorCorrectedImages/CCF" + str(i) + ".jpg"
        cc_image = cv2.imread(image)

        f = facial_features.detect_facial_landmarks(cc_image)
        img_arr = [f[3], f[4], f[5]]
        l_avg, a_avg, b_avg = facial_features.total_under_tone(img_arr)
        """
        threshold_value = 70 # 100
        mask = facial_features.get_hair_mask(cc_image, threshold_value)

        l_hair, a_hair, b_hair = facial_features.get_lab_color_space(mask)
        if l_hair > 50:
            # REDO THE HAIR MASK!! - was at 190
            threshold_value = 120
            mask = facial_features.get_hair_mask(cc_image, threshold_value)
            l_hair, a_hair, b_hair = facial_features.get_lab_color_space(mask)

        top_3_colors = facial_features.get_top_color(
            mask, num_colors=3, value_threshold=25, in_BGR=1
        )
        """
        top_3_colors, l_hair, a_hair, b_hair, hair_mask = facial_features.get_hair_values(cc_image, in_BGR=1)

        # Lab values,
        data.append(
            (
                (l_avg, a_avg, b_avg),
                cc_image,
                hair_mask,
                (l_hair, a_hair, b_hair),
                top_3_colors,
            )
        )

        print(counter)
        counter += 1

    # L - x[3][0] || a - x[3][1] || b - x[3][2]
    sorted_arr = sorted(data, key=lambda x: x[3][0], reverse=True)

    for ar in sorted_arr:
        # ORIGINAL IMAGE
        img = ar[1]
        _, buffer = cv2.imencode(".jpg", img)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(
            f'<td><img src="data:image/jpeg;base64,{img_str}" width="{img.shape[1]//8}" height="{img.shape[0]//8}"></td>'
        )

        img = ar[2]
        _, buffer = cv2.imencode(".jpg", img)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(
            f'<td><img src="data:image/jpeg;base64,{img_str}" width="{img.shape[1]//4}" height="{img.shape[0]//4}"></td>'
        )

        for i in range(0, 3):
            rgb_color = ar[4][i]

            img = Image.new("RGB", (50, 50), rgb_color)
            img_np = np.array(img)

            _, buffer = cv2.imencode(".jpg", img_np)
            img_str = base64.b64encode(buffer).decode("utf-8")
            rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')

        # rows.append(f'<td>(L,a,b) = ({ar[0][0]:.2f}, {ar[0][1]:.2f}, {ar[0][2]:.2f})</td>')
        rows.append(
            f"<td>HAIR: (L,a,b) = ({ar[3][0]:.2f}, {ar[3][1]:.2f}, {ar[3][2]:.2f})</td>"
        )
        rows.append(f"<td> Top 3 colors: = ({ar[4]})</td>")

        # bgr_val = ar[0]
        # rows.append(f'<td>(R,G,B) = ({bgr_val[2]},{bgr_val[1]},{bgr_val[0]})</td>')
        rows.append("</tr><tr>")

    html = html_template.format(rows="\n".join(rows))
    # Write the HTML to a file
    filename = filename + ".html"
    with open(filename, "w") as f:
        f.write(html)


def display_all_features_HTML(img_str: str, counter: int, ours: bool, color_correct: bool, in_BGR: int) -> List[str]:
    """
    This function generates the rows for a single image that will appear in the HTML file containing all feature information.
    What is displayed on the HTML file: Original image, forehead, left cheek, right cheek images, skin Lab value, 
    left eye, right eye, left eye iris box, example of RGB value, eye Lab values, eye RGB value,
    hair image, 3 examples boxes of the top 3 RGB values, hair Lab value and top 3 RGB values. 

    Parameters:
    ----------
    img_str : str
        String representing the file of the image
    ours : bool
        True - if the image comes from the images we took, False - if comes from Chicago Database
    color_correct : bool
        True - if the image needs to be color corrected
    in_BGR : int
        A flag indicating indicate whether the incoming mask is in BGR format

    Returns:
    ----------
    rows : List[str]
        The rows HTML code needed to create the HTML file
    """
    rows = []
    data = facial_features.facial_features_and_values(
        img_str, ours, color_correct, in_BGR
    )

    # original_image = data['original_image']
    color_corrected_image = data["color_corrected_image"]

    _, buffer = cv2.imencode(".jpg", color_corrected_image)
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(f"<div>Color Corrected Image: {counter}</div>")
    rows.append(
        f'<div><img src="data:image/jpeg;base64,{img_str}" width="{color_corrected_image.shape[1]//4}" height="{color_corrected_image.shape[0]//4}"></div>'
    )
    rows.append("</tr><tr>")

    # SKIN (FOREHEAD, LEFT CHEEK, RIGHT CHEEK)
    # left_cheek, right_cheek, forehead
    rows.append("<table>")
    _, buffer = cv2.imencode(".jpg", data["forehead"])
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(f"<td>Forehead -- Left cheek -- Right cheek</td>")
    rows.append("</tr><tr>")
    rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')
    _, buffer = cv2.imencode(".jpg", data["cheek_left"])
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')
    _, buffer = cv2.imencode(".jpg", data["cheek_right"])
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')
    skin_lab = data["skin_lab"]
    rows.append(
        f"<td>(L,a,b) = ({skin_lab[0]:.2f}, {skin_lab[1]:.2f}, {skin_lab[2]:.2f})</td>"
    )
    rows.append("</tr><tr>")
    rows.append("</table>")

    # EYES

    rows.append("<table>")
    _, buffer = cv2.imencode(".jpg", data["eye_left"])
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(f"<td>Left eye -- Right eye</td>")
    rows.append("</tr><tr>")
    rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')
    _, buffer = cv2.imencode(".jpg", data["eye_right"])
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')

    img = data["iris_mask"]
    _, buffer = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(
        f'<td><img src="data:image/jpeg;base64,{img_str}" width="{4*img.shape[1]}" height="{4*img.shape[0]}"></td>'
    )

    eye_RGB = data["eye_RGB"]
    flipped_RGB = eye_RGB[::-1]
    img = Image.new("RGB", (50, 50), flipped_RGB)
    img_np = np.array(img)

    _, buffer = cv2.imencode(".jpg", img_np)
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')

    eye_lab = data["eye_lab"]
    rows.append(
        f"<td>(L,a,b) = ({eye_lab[0]:.2f}, {eye_lab[1]:.2f}, {eye_lab[2]:.2f})</td>"
    )
    rows.append(
        f"<td>(R,G,B) = ({eye_RGB[0]:.2f}, {eye_RGB[1]:.2f}, {eye_RGB[2]:.2f})</td>"
    )

    rows.append("</tr><tr>")
    rows.append("</table>")

    # HAIR
    img = data["hair_mask"]
    _, buffer = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(buffer).decode("utf-8")
    rows.append(f"<td>Hair</td>")
    rows.append("</tr><tr>")
    rows.append(
        f'<td><img src="data:image/jpeg;base64,{img_str}" width="{img.shape[1]//2}" height="{img.shape[0]//2}"></td>'
    )

    hair_colors = data["hair_colors"]
    for i in range(0, 3):
        rgb_color = hair_colors[i]
        flipped_RGB = rgb_color[::-1]
        img = Image.new("RGB", (50, 50), flipped_RGB)
        img_np = np.array(img)

        _, buffer = cv2.imencode(".jpg", img_np)
        img_str = base64.b64encode(buffer).decode("utf-8")
        rows.append(f'<td><img src="data:image/jpeg;base64,{img_str}"></td>')

    hair_lab = data["hair_lab"]
    rows.append(
        f"<td>(L,a,b) = ({hair_lab[0]:.2f}, {hair_lab[1]:.2f}, {hair_lab[2]:.2f})</td>"
    )
    top_3_colors = data["hair_colors"]
    rows.append(f"<td> Top 3 colors: = ({top_3_colors})</td>")

    rows.append("</tr><tr>")
    rows.append("</table>")
    return rows


def create_HTML_file_all_features_specific_our_photos(
    filename: str, foldername: str, img_list: List[str], ours=True, color_correct=True, in_BGR=1
) -> None:
    """
    This function generates an HTML file from images in the specified folded from the photos we took.
    The file will contain all of the information from the function display_all_features_HTML for each image and append all information to the same HTML file

    Parameters:
    ----------
    filename : str
        Name of HTML file you want to create and write to 
    foldername : str
        Desired folder the images will be in
    img_list: List[str]
        List of strings representing the file of the image
    ours : bool
        True - if the image comes from the images we took, False - if comes from Chicago Database
    color_correct : bool
        True - if the image needs to be color corrected
    in_BGR : int
        A flag indicating indicate whether the incoming mask is in BGR format

    Returns:
    ----------
    rows : List[str]
        The rows HTML code needed to create the HTML file
    """

    # Create an HTML template for displaying the images
    html_template = """
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
    """

    # Generate HTML code for each row of images
    rows = []
    counter = 0
    # Process the files in order
    for img_str in img_list:
        img_str = foldername + img_str
        img_HTML_rows = display_all_features_HTML(
            img_str, counter, ours, color_correct, in_BGR=1
        )
        rows += img_HTML_rows
        print(counter)
        counter += 1

    html = html_template.format(rows="\n".join(rows))
    # Write the HTML to a file
    filename = filename + ".html"
    with open(filename, "w") as f:
        f.write(html)


def create_HTML_file_all_features_specific(
    filename: str, img_str: str, ours: bool=False, color_correct: bool=True
) -> None:
    """
    This function generates an HTML file from a selected image in the Chicago Database photos
    The file will contain all of the information from the function display_all_features_HTML for the image in an HTML file

    Parameters:
    ----------
    filename : str
        Name of HTML file you want to create and write to 
    img_str
        Strings representing the file of the image
    ours : bool
        True - if the image comes from the images we took, False - if comes from Chicago Database
    color_correct : bool
        True - if the image needs to be color corrected
    """

    # Create an HTML template for displaying the images
    html_template = """
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
    """

    # Generate HTML code for each row of images
    rows = []
    counter = 0
    img_HTML_rows = display_all_features_HTML(
        img_str, counter, ours, color_correct, in_BGR=1
    )
    rows += img_HTML_rows

    html = html_template.format(rows="\n".join(rows))
    # Write the HTML to a file
    filename = filename + ".html"
    with open(filename, "w") as f:
        f.write(html)


def create_HTML_file_all_features(
    filename: str, start_img: int, end_img: int, ours: bool=False, color_correct: bool=True
) -> None:
    """
    This function generates an HTML file from images in the specified folded from the Chicago Database
    The file will contain all of the information from the function display_all_features_HTML for each image and append all information to the same HTML file

    Parameters:
    ----------
    filename : str
        Name of HTML file you want to create and write to 
    start_img : int
        The index of the first image to be processed.
    end_img : int 
        The index of the last image to be processed (not including)  
    ours : bool
        True - if the image comes from the images we took, False - if comes from Chicago Database
    color_correct : bool
        True - if the image needs to be color corrected
    """
    # Create an HTML template for displaying the images
    html_template = """
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
    """

    # Generate HTML code for each row of images
    rows = []
    counter = start_img

    # Specify the directory path
    dir_path = "ChicagoFaceDatabaseImages"
    # dir_path = "OurPhotos"

    # Get a list of all the files in the directory
    file_list = os.listdir(dir_path)

    # Sort the list of files in alphabetical order
    sorted_files = sorted(file_list)
    sorted_files = sorted_files[start_img:end_img]

    # Process the files in order
    for file_name in sorted_files:
        img_str = "ChicagoFaceDatabaseImages/" + file_name

        img_HTML_rows = display_all_features_HTML(
            img_str, counter, ours, color_correct, 0
        )
        rows += img_HTML_rows

        print(counter)
        counter += 1

    html = html_template.format(rows="\n".join(rows))
    # Write the HTML to a file
    filename = filename + ".html"
    with open(filename, "w") as f:
        f.write(html)



# Test that these function work 
"""
directory_name = "test_HTML_files/"
if not os.path.exists(directory_name):
    os.makedirs(directory_name)

display_skin_info_HTML(directory_name + "test_skin_chicago", 0, 1)
display_eye_info_HTML(directory_name + "test_eyes_chicago", 0, 1)
display_hair_info_HTML(directory_name + "test_hair_chicago", 0, 1)


# Use this function to look through all the images in OurPhotos
img_list = ["DSC06469.JPG", "DSC06471.JPG", "DSC06473.JPG", "DSC06474.JPG","DSC06477.JPG",
           "DSC06479.JPG", "DSC06481.JPG", "DSC06483.JPG", "DSC06484.JPG", "DSC06488.JPG",
            "DSC06489.JPG", "DSC06491.JPG", "DSC06494.JPG"]
create_HTML_file_all_features_specific_our_photos(directory_name + "test_our_photos", "OurPhotos/", img_list[0:1], True)


img_str = "OurPhotos/DSC06469.JPG"
create_HTML_file_all_features_specific(directory_name + "test_our_photo_specific", img_str,  1)


# HTML filename, start img #, end img #, color_correct True or False
create_HTML_file_all_features(directory_name + "test_chicago_photos", 0, 1)
"""





