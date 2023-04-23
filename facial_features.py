import dlib
import cv2
import sys
import numpy as np
import glob
from PIL import Image



def detect_facial_landmarks(img_path):
    # Load the image and convert it to grayscale
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()

    # Create a shape predictor object to locate facial landmarks
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    faces = detector(gray)


    # Loop over each detected face and get the facial landmarks
    for face in faces:
        landmarks = predictor(gray, face)

        # Loop over the 68 facial landmarks and draw them on the image
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            #cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  # Draw a red dot at the landmark location
            #cv2.putText(image, str(i+1), (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

        facial_features = []
        # EYES
        leftEyeLandmarks = [36, 37, 38, 39, 40, 41]
        leftEyeImage = createMask(landmarks, leftEyeLandmarks, image)
        #eye_color = find_iris(leftEyeImage)
        facial_features.append(leftEyeImage)

        rightEyeLandmarks = [42, 43, 44, 45, 46, 47]
        rightEyeImage = createMask(landmarks, rightEyeLandmarks, image)
        facial_features.append(rightEyeImage)

        # LIPS
        #lipsLandmarks = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,60, 61, 62, 63, 64, 65, 66, 67]
        lipsLandmarks = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,]

        lipsImage = createMask(landmarks, lipsLandmarks, image)
        facial_features.append(lipsImage)

        # CHEEKS
        leftCheekLandmarks = [2,3,4,48]
        leftCheekImage = createMask(landmarks, leftCheekLandmarks, image, 1)
        facial_features.append(leftCheekImage)
        rightCheekLandmarks = [14,13,12,54]
        rightCheekImage = createMask(landmarks, rightCheekLandmarks, image, 2)
        facial_features.append(rightCheekImage)

        # FOREHEAD
        foreheadImage = createMask(landmarks, [], image, 3)
        facial_features.append(foreheadImage)

    return facial_features

def createMask(landmarks, specific_landmarks, image, swatch = 0):
    new_point = None
    # the left cheek, right cheek, and forehead need added points 
    # left cheek
    if swatch == 1:
        x = landmarks.part(48).x 
        y = landmarks.part(2).y
        new_point = (x, y)

    # right cheek
    if swatch == 2:
        x = landmarks.part(54).x 
        y = landmarks.part(14).y
        new_point = (x, y)

    # forehead
    foreheadPoints = np.empty((0, 2))
    if swatch == 3:
        leftMostY = landmarks.part(19).x
        rightMostY = landmarks.part(24).x
        bottom = min(landmarks.part(19).y,landmarks.part(24).y)
        top = bottom-60 # tbd
        foreheadPoints = np.concatenate((foreheadPoints, np.array([(leftMostY,bottom)])))
        foreheadPoints = np.concatenate((foreheadPoints, np.array([(rightMostY,bottom)])))
        foreheadPoints = np.concatenate((foreheadPoints, np.array([(rightMostY,top)])))
        foreheadPoints = np.concatenate((foreheadPoints, np.array([(leftMostY,top)])))
        foreheadPoints = foreheadPoints.astype(int)



    # Create mask
    points = np.empty((0, 2))
    points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in specific_landmarks])
    if new_point is not None and swatch !=3:
        points = np.concatenate((points, np.array([new_point])))
    if swatch == 3:
        points = foreheadPoints.copy()

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [points], -1, 255, -1, cv2.LINE_AA)

    # Apply mask to original image to get left eye image
    maskedImage = cv2.bitwise_and(image, image, mask=mask)

    #return maskedImage

    # Crop the image to the masked part
    x, y, w, h = cv2.boundingRect(mask)
    croppedImage = maskedImage[y:y+h, x:x+w]
    return croppedImage

def find_iris(eyeMask):
    gray_eyeMask = cv2.cvtColor(eyeMask, cv2.COLOR_BGR2GRAY)
    
    # Get bounding box of eyeMask
    x, y, w, h = cv2.boundingRect(gray_eyeMask)
    
    # Make iris section bounding box
    y_mod = round(y + (3.25*(h/6)))
    x_mod = round(x + (1.5*(w/4)))
    
    x_diff = round(x_mod + (w/4))
    y_diff = round(y_mod + .75*(h/6))
    
    # Get iris section from eyeMask
    irisMask = eyeMask[y_mod:y_diff, x_mod:x_diff, :]

    # Convert irisMask to Lab color space
    lVal, aVal, bVal = getLabColorSpace(irisMask)
    #print("L: " + str(lVal) + " a: " + str(aVal) + " b: " + str(bVal))
    
    # Get most prominent color in iris using color histogram
    hist = cv2.calcHist([irisMask], [0,1,2], None, [256,256,256], [0,256,0,256,0,256])
    hist_flatten = hist.flatten()
    max_color_ind = np.argmax(hist_flatten)
    bgr = np.unravel_index(max_color_ind, hist.shape)
    eye_color = (bgr[0], bgr[1], bgr[2])
    
    # Draw rectangle on image
    # cv2.rectangle(eyeMask, (x_mod, y_mod), (round(x_mod + (w/4)), round(y_mod + (h/6))), (0, 0, 255), 1)
    
    return (eye_color, lVal, aVal, bVal, irisMask)
    #return eye_color

def getLabColorSpace(mask):
    # Convert irisMask to Lab color space
    lab_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2LAB)

    # Split into L, a, and b channels
    L_channel, a_channel, b_channel = cv2.split(lab_mask)
    lValue = np.mean(L_channel)
    aValue = np.mean(a_channel)
    bValue = np.mean(b_channel)

    return (lValue, aValue, bValue)
    
def under_tone(img):
    # Convert RGB to LAB
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Split into L, a, and b channels
    L_channel, a_channel, b_channel = cv2.split(img_lab)

    # Use mask to extract non-black pixels from a and b channels
    a_values = a_channel[mask != 0]

    # Calculate average a and b values
    a_avg = np.mean(a_values)


    return a_avg

# Return the undertone using the 3 swatches - left cheek, right cheek, and forehead
# higher a* value would indicate a cooler or pinker undertone, while a lower a* value would indicate a warmer or yellower undertone.
def total_under_tone(img_path):
    total_under_tone_val = 0
    f = detect_facial_landmarks(img_path)  
    # 1.leftCheek, 2.rightCheek, 3.forehead
    for i in range(0,3):
        img = f[i+3]      
        underTone = under_tone(img)
        total_under_tone_val +=  underTone

    avgUndertone = total_under_tone_val/3

    # Threshold for cool, neutral, warm
    if avgUndertone > 129:
        print("Cool")
    elif avgUndertone >= 128:
        print("Neutral")
    else:
        print("Warm")
    
    return avgUndertone

    
def getHair(img_path):
    image = cv2.imread(img_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a median blur to reduce noise
    blurred_image = cv2.medianBlur(gray_image, 5)

    # Convert the grayscale image to a 3-channel image
    blurred_image_3ch = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)

    # Set the threshold and new_value
    threshold = 1
    new_value = (128, 128, 128)

    # Find the seed point in the middle of the image at the top and apply the floodFill function
    middle_column = blurred_image.shape[1] // 2
    for y in range(blurred_image.shape[0]):
        if blurred_image[y, middle_column] < new_value[0] - threshold:
            seed_point = (middle_column, y)
            mask = np.zeros((blurred_image_3ch.shape[0] + 2, blurred_image_3ch.shape[1] + 2), dtype=np.uint8)
            lo_diff = (threshold,) * 3
            up_diff = (threshold,) * 3
            cv2.floodFill(blurred_image_3ch, mask, seed_point, new_value, lo_diff, up_diff, cv2.FLOODFILL_FIXED_RANGE)
            break

    # Convert the 3-channel image back to a single channel image
    blurred_image_filled = cv2.cvtColor(blurred_image_3ch, cv2.COLOR_BGR2GRAY)

    # Set the y-coordinate limit
    y_limit = 350

    # Create a region of interest (ROI) above the y-coordinate limit
    roi = blurred_image_filled[:y_limit, :]

    # Create the hair mask using the ROI
    _, hair_mask_roi = cv2.threshold(roi, new_value[0] - threshold, 255, cv2.THRESH_BINARY)

    hair_mask_roi_inv = cv2.bitwise_not(hair_mask_roi)

    # Create a full-size hair mask with the same dimensions as the original image
    hair_mask = np.zeros_like(blurred_image_filled)
    hair_mask[:y_limit, :] = hair_mask_roi_inv

    # hair_mask_inv = cv2.bitwise_not(hair_mask)

    # Create the masked image
    masked_image = cv2.bitwise_and(image, image, mask=hair_mask)
    lVal, aVal, bVal = getLabColorSpace(masked_image)
    # Compute the average pixel value of the detected hair region in the color image
    average_hair_value_color = cv2.mean(image, mask=hair_mask)

    resized_image = cv2.resize(image, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
    
    resized_hair_mask = cv2.resize(hair_mask, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
    # masked_image = cv2.bitwise_and(image, image, mask=hair_mask)
    resized_masked_image = cv2.resize(masked_image, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)


    # cv2.imshow('Original Image', resized_image)
    # # cv2.imshow('Hair Mask', resized_hair_mask)
    # cv2.imshow('Masked Image', resized_masked_image)
    # # resized_hair_swatch = cv2.resize(hair_swatch, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
    # # cv2.imshow('Hair Swatch', hair_swatch)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #return average_hair_value_color
    return (average_hair_value_color, lVal, aVal, bVal,resized_masked_image)

def flood_fill(image, seed_point, threshold, new_value):
    h, w = image.shape
    visited = np.zeros_like(image, np.uint8)
    stack = [seed_point]

    while stack:
        x, y = stack.pop()
        if visited[y, x]:
            continue
        visited[y, x] = 1

        if abs(image[y, x] - image[seed_point]) <= threshold:
            image[y, x] = new_value

            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    stack.append((nx, ny))




# # TEST THE FUNCTION 1
img_path = "ChicagoFaceDatabaseImages/CFD-AF-202-122-N.jpg"
# facial_features = detect_facial_landmarks(img_path)

"""
a_value = total_under_tone(img_path)
print(a_value)

cv_img = cv2.imread(img_path)
scale = 0.5
resized_img = cv2.resize(cv_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
# og resized img
cv2.imshow("Resized Image", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""



'''
counter = 0
for featureImage in facial_features:
    #if counter == 3:
        #under_tone(featureNames[counter])
    cv2.imshow(featureNames[counter], featureImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    counter+=1
'''

    


# # TEST FOR IRIS DETECTION
# for image in glob.glob("./ChicagoFaceDatabaseImages/*.jpg"):
#     detect_facial_landmarks(image)