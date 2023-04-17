import numpy as np
import cv2
import glob

def color_correct():
    count = 0
    for image in glob.glob("./ChicagoFaceDatabaseImages/*.jpg"):
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
        filename = f"./ColorCorrectedImages/CCF{count}.jpg"
        cv2.imwrite(filename, final_img)
        count += 1
        # cv2.imshow("Color Corrected Image", final_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()