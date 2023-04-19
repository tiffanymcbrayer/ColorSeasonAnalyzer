import dlib
import cv2
import sys
import numpy as np


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

    





# TEST THE FUNCTION 
img_path = "ChicagoFaceDatabaseImages/CFD-AF-202-122-N.jpg"
facial_features = detect_facial_landmarks(img_path)

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

    

