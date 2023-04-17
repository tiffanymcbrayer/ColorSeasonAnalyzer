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
        lipsLandmarks = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
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
        top = bottom-100 # tbd
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

    return maskedImage



# TEST THE FUNCTION 
img_path = "ChicagoFaceDatabaseImages/CFD-AF-202-122-N.jpg"
facial_features = detect_facial_landmarks(img_path)

featureNames = ["left eye", "right eye", "lips", "left cheek", "right cheek", "forehead"]

counter = 0
for featureImage in facial_features:
    cv2.imshow(featureNames[counter], featureImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    counter+=1
