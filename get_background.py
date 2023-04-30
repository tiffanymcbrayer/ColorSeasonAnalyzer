import numpy as np
import cv2
import glob
import numpy.typing as npt


def get_avg_bg(cv_img: np.ndarray) -> np.ndarray:
    """
    This function acts as a color corrector for our own visual data separate from the general color corrector.
    Since we want our visual data to mimic the dataset of photos as much as possible, we have another color corrector getting the hue and saturation 
    values of the background of the dataset and comparing it to our photos. We make up for the difference in hue and saturation in our
    photos. 
    
    Parameters:
    ----------
    cv_img : np.ndarray 
        Original photo from our own visual data.

    Returns:
    -------
    correct_img : np.ndarray
        A color corrected version of our photo based on the dataset.
    """
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
    new_img = cv2.merge((corrected_h, corrected_s, hsv_img[:, :, 2]))
    corrected_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)
    scalex = 859 / corrected_img.shape[0]
    scaley = 1222 / corrected_img.shape[1]
    correct_img = cv2.resize(
        corrected_img, None, fx=scalex, fy=scaley, interpolation=cv2.INTER_AREA
    )

    return correct_img

