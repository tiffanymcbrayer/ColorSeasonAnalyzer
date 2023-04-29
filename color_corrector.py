import numpy as np
import cv2
import glob
import numpy.typing as npt

def color_corrector(image) -> None:
    """This function is to color correct all original photos from the Chicago Face Database. This is mainly using the algorithm from
    https://stackoverflow.com/questions/70876252/how-to-do-color-cast-removal-or-color-coverage-in-python on how to remove color cast
    from photos. To remove color cast you must convert the image from BGR to HSV color space and then flip the hue channel
    to its complement. Then you merge the new hue channel back with the original S and V channels and convert back to BGR. Then you get
    the average color of that reconverted BGR image and make a new image with that average color. You blend the two images together 50-50
    and normalize the photo."""
    # resizing img to fit in window
    cv_img = cv2.imread(image)
    scale = 0.5
    resized_img = cv2.resize(cv_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    # og resized img
    # cv2.imshow("Resized Image", resized_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # converting to HSV and splitting channels
    h, s, v = cv2.split(hsv_img)
    # flippig hue to be its complement
    h_complement = (h + 90) % 180
    # merge hue complement with original saturation and value
    comp_hsv_img = cv2.merge([h_complement, s, v])
    comp_bgr_img = cv2.cvtColor(comp_hsv_img, cv2.COLOR_HSV2BGR)
    avg_color = cv2.mean(comp_bgr_img)
    constant_color = tuple(map(int, avg_color[:3]))
    avg_img = np.full_like(resized_img, constant_color)
    # blending constant color image and og image together
    blended_img = cv2.addWeighted(resized_img, 0.5, avg_img, 0.5, 0)
    # normalizing image
    final_img= cv2.normalize(blended_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # cv2.imshow("Color Corrected Image", final_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return final_img
        
def color_correct_all():
    count = 0
    for image in glob.glob("./ChicagoFaceDatabaseImages/*.jpg"):
        cc_img = color_corrector(image)
        filename = f"./ColorCorrectedImages/CCF{count}.jpg"
        cv2.imwrite(filename, cc_img)
        count += 1