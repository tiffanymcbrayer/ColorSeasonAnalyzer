import numpy as np
import cv2
import glob
import numpy.typing as npt


def get_avg_bg(cv_img):
    # base hue and saturation using hsv color space from a square of background in our original dataset
    base_h = 101
    base_s = 17
    # cv_img = cv2.imread(image)
    hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

    # hue and saturation from incoming photo
    background_h = int(np.mean(hsv_img[:100, :100, 0].flatten()))
    background_s = int(np.mean(hsv_img[:100, :100, 1].flatten()))

    # difference in h and s of dataset and own visual data
    h_diff = base_h - background_h
    s_diff = base_s - background_s

    # add difference
    corrected_s = cv2.add(hsv_img[:, :, 1], s_diff, dtype=cv2.CV_8UC1)
    corrected_h = cv2.add(hsv_img[:, :, 0], h_diff, dtype=cv2.CV_8UC1)

    # remerged image and resized
    corrected_img = cv2.merge((corrected_h, corrected_s, hsv_img[:, :, 2]))
    bgr_img = cv2.cvtColor(corrected_img, cv2.COLOR_HSV2BGR)
    scalex = 859 / bgr_img.shape[0]
    scaley = 1222 / bgr_img.shape[1]
    resized_img = cv2.resize(
        bgr_img, None, fx=scalex, fy=scaley, interpolation=cv2.INTER_AREA
    )
    # cv2.imshow("img", resized_img)
    # cv2.imshow("cc", cv_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return resized_img


# # # test
# count = 0
# for image in glob.glob("./OurPhotos/CCF*.jpg"):
#     cc_img = get_avg_bg(image)
#     filename = f"./OurPhotos/BCCF{count}.jpg"
#     cv2.imwrite(filename, cc_img)
#     count += 1
# img_path = "./CCDSC06469.JPG"
# new_img = get_avg_bg(img_path)
# cv2.imwrite("./2BCCDSC06469.JPG",new_img)

# test2
# img_path = "./ColorCorrectedImages/CCF0.JPG"
# get_avg_bg(img_path)
# cv2.imwrite("./BCCF0.JPG",new_img)
