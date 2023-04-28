import dlib
import cv2
import sys
import numpy as np
import glob
from PIL import Image
import color_correct
from skimage.feature import local_binary_pattern


#def detect_facial_landmarks(img_path):
def detect_facial_landmarks(image):
    # Load the image and convert it to grayscale
    #image = cv2.imread(img_path)
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

# def getLabColorSpace(mask):
#     # Convert irisMask to Lab color space
#     lab_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2LAB)

#     # Split into L, a, and b channels
#     L_channel, a_channel, b_channel = cv2.split(lab_mask)
#     lValue = np.mean(L_channel)
#     aValue = np.mean(a_channel)
#     bValue = np.mean(b_channel)

#     return (lValue, aValue, bValue)
    
# Undertone from Lab color space 
#def under_tone(img):
def getLabColorSpace(img):
    # Convert RGB to LAB
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Split into L, a, and b channels
    L_channel, a_channel, b_channel = cv2.split(img_lab)

    # Use mask to extract non-black pixels from a and b channels
    l_values = L_channel[mask != 0]
    a_values = a_channel[mask != 0]
    b_values = b_channel[mask != 0]


    # Calculate average a and b values
    l_avg = np.mean(l_values)
    a_avg = np.mean(a_values)
    b_avg = np.mean(b_values)


    return (l_avg, a_avg, b_avg)

# Return the undertone using the 3 swatches - left cheek, right cheek, and forehead
# higher a* value would indicate a cooler or pinker undertone, while a lower a* value would indicate a warmer or yellower undertone.
def total_under_tone(imgArr):
    l_tot = 0
    a_tot = 0
    b_tot = 0
    # 1.leftCheek, 2.rightCheek, 3.forehead
    for i in range(0,3):
        img = imgArr[i]      
        l, a, b = getLabColorSpace(img)
        l_tot += l
        a_tot += a
        b_tot += b
  
    l_avg = l_tot/3
    a_avg = a_tot/3
    b_avg = b_tot/3

    """
    # Threshold for cool, neutral, warm
    if avgUndertone > 129:
        print("Cool")
    elif avgUndertone >= 128:
        print("Neutral")
    else:
        print("Warm")
    """

    return (l_avg, a_avg, b_avg)

    
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
    y_limit = 50

    (average_hair_value_color, lVal, aVal, bVal) = ((0.0, 0.0, 0.0, 0.0), 0.0, 128.0, 128.0)

    while (average_hair_value_color, lVal, aVal, bVal) == ((0.0, 0.0, 0.0, 0.0), 0.0, 128.0, 128.0):
        y_limit += 50
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
        # resized_hair_swatch = cv2.resize(hair_swatch, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
        # cv2.imshow('Hair Swatch', hair_swatch)

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


def display_all_hair():
    for image in glob.glob("./ColorCorrectedImages/*.jpg"):
        h = getHair(image)
        print(h)


    



# # TEST FOR IRIS DETECTION
# for image in glob.glob("./ChicagoFaceDatabaseImages/*.jpg"):
#     detect_facial_landmarks(image)


def find_intersection(line1_point1, line1_point2, line2_point1, line2_point2):
    # Extract coordinates from the input points
    x1, y1 = line1_point1
    x2, y2 = line1_point2
    x3, y3 = line2_point1
    x4, y4 = line2_point2
    
    # Compute the slopes of the lines
    m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
    m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else float('inf')
    
    # Compute the y-intercepts of the lines
    b1 = y1 - m1 * x1 if x2 != x1 else x1
    b2 = y3 - m2 * x3 if x4 != x3 else x3
    
    # If the lines are parallel, return None
    if m1 == m2:
        return None
    
    # Compute the intersection point
    x_int = (b2 - b1) / (m1 - m2)
    y_int = m1 * x_int + b1
    
    # Return the intersection point
    return [int(x_int), int(y_int)]



# Right now getting top 3 colors in a mask 
def get_top_color(mask, num_colors=3, value_threshold=10, inBGR=1):
    # Get top 3 most prominent colors in iris using color histogram, excluding black color
    hist = cv2.calcHist([mask], [0,1,2], None, [256,256,256], [0,256,0,256,0,256])
    hist_flatten = hist.flatten()
    hist_flatten[0] = 0 # set the count of black color to 0
    max_color_inds = np.argsort(hist_flatten)[::-1] # get the indices of the top colors
    top3colors = []
    for ind in max_color_inds:
        bgr = np.unravel_index(ind, hist.shape)
        rgb = (bgr[2], bgr[1], bgr[0])
        if len(top3colors) == 0:
            top3colors.append(rgb)
        else:
            diff = np.abs(np.array(top3colors) - np.array(rgb))
            if np.all(diff >= value_threshold):
                top3colors.append(rgb)
        if len(top3colors) == num_colors:
            break

    if inBGR:
        top3colors = [c[::-1] for c in top3colors]
    return top3colors






# https://github.com/codeniko/shape_predictor_81_face_landmarks

def get_hair_mask(image, threshold_value):
    # Load image
    #image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize dlib's face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

    # Detect faces in the image
    faces = detector(gray)

    # Create a binary mask of the hair region
    mask = np.zeros_like(gray)
    for face in faces:
        # Get the facial landmarks for the face
        landmarks = predictor(gray, face)

        # Get the points corresponding to the forehead and the hairline
       # forehead_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 80)])
        forehead_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in [75, 76, 68, 69]])
        forehead_points = np.vstack([forehead_points, [landmarks.part(75).x, landmarks.part(68).y]])
        center_x = (landmarks.part(75).x + landmarks.part(68).x + landmarks.part(75).x) / 3
        center_y = (landmarks.part(75).y + landmarks.part(68).y + landmarks.part(69).y) / 3
        forehead_points = np.vstack([forehead_points, [center_x,center_y]])

        forehead_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0,80)])
        forehead_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in [72, 73, 79, 74]])


        x,y,_ = image.shape

        intersection = find_intersection((landmarks.part(68).x, landmarks.part(68).y),
                           (landmarks.part(69).x, landmarks.part(69).y), 
                           (landmarks.part(72).x, landmarks.part(72).y), 
                           (landmarks.part(73).x, landmarks.part(73).y))
        
        #
        # all points 
        forehead_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in [75, 76, 68, 69]]) 
        forehead_points = np.vstack([forehead_points, [intersection]])
        leftSide = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in [72, 73, 79, 74]]) 
        forehead_points = np.vstack([forehead_points, leftSide])
        forehead_points = np.vstack([forehead_points, [y-5, landmarks.part(74).y]])
        forehead_points = np.vstack([forehead_points, [y-5, landmarks.part(74).y]])
        forehead_points = np.vstack([forehead_points, [y-5,0]])
        forehead_points = np.vstack([forehead_points, [0, 0]])
        forehead_points = np.vstack([forehead_points, [0, landmarks.part(75).y]])

        
        
        hairPoints = [75, 76, 68, 69, 70]
        hairline_points = np.array([(landmarks.part(i).y, landmarks.part(i).x) for i in [72, 73, 79, 74]])


        # Create a convex hull of the points to get the hair region
       # hair_points = np.concatenate((forehead_points, hairline_points[::-1]), axis=0)
        #hull = cv2.convexHull(forehead_points)
        #cv2.fillConvexPoly(mask, hull, 255)

        # Draw circles on the original image at the location of each point
        # for i, point in enumerate(forehead_points):
        #     x, y = point
        #     cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

        #     cv2.putText(image, str(i), (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [forehead_points], -1, 255, -1, cv2.LINE_AA)

        # Apply mask to original image to get left eye image
        maskedImage = cv2.bitwise_and(image, image, mask=mask)

        #return maskedImage

        # Crop the image to the masked part
        x, y, w, h = cv2.boundingRect(mask)
        croppedImage = maskedImage[y:y+h, x:x+w]
    #---------------------------------------------------

    img = croppedImage

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a median filter to remove noise
    gray = cv2.medianBlur(gray, 5)

    # Apply a threshold to create a binary image
    ret, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to remove noise and smooth out edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.erode(binary, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Create a black mask with the same size as the image
    mask = np.zeros_like(img[:,:,0])

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(h) / w
        area = cv2.contourArea(c)
        #if aspect_ratio > 2 and area > 100 and area < 1000:
        cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)

    mask = cv2.bitwise_not(mask)

    result = cv2.bitwise_or(img, img, mask=mask)
    
    return result



def facial_features_and_values(img_str):
    original_image = cv2.imread(img_str)

    # COLOR CORRECT THE IMAGE 
    image = color_correct.color_correct(img_str)


    # Detect all facial features
    f = detect_facial_landmarks(image)   
    eyeLeft = f[0]
    eyeRight = f[1]
    cheekLeft = f[3]
    cheekRight = f[4]
    forehead = f[5]
    skinArr = [cheekLeft, cheekRight, forehead]


    # SKIN undertone Lab values 
    l_avg_skin, a_avg_skin, b_avg_skin = total_under_tone(skinArr)

 
    # EYES 
    eye_color, l_avg_eye, a_avg_eye, b_avg_eye, irisMask = find_iris(eyeLeft)
    eye_color = (eye_color[2], eye_color[1], eye_color[0])
    print(eye_color)
    
    

    # HAIR
    threshold_value = 100
    hairMask = get_hair_mask(image, threshold_value)
    l_hair, a_hair, b_hair  = getLabColorSpace(hairMask)
    if l_hair > 70:
        # REDO THE HAIR MASK!!
        threshold_value = 190
        hairMask = get_hair_mask(image, threshold_value)
        l_hair, a_hair, b_hair  = getLabColorSpace(hairMask)

    top3colors = get_top_color(hairMask)

    img = Image.new('RGB', (50, 50), eye_color)
    img_np = np.array(img)

    # cv2.imshow("HAIR", irisMask)
    # cv2.imshow("Image", img_np)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    data = {'original_image': original_image, 
               'color_corrected_image': image,
               'eyeLeft': eyeLeft,
               'eyeRight': eyeRight,
               'cheekLeft': cheekLeft,
               'cheekRight': cheekRight,
               'forehead': forehead,
               'skinLab': (l_avg_skin, a_avg_skin, b_avg_skin),
               'eyeRGB': eye_color,
               'eyeLab': (l_avg_eye, a_avg_eye, b_avg_eye),
               'irisMask': irisMask,
               'hairLab': (l_hair, a_hair, b_hair),
               'hairColors': top3colors,
               'hairMask': hairMask
               }
 


    return data

img_str = "ChicagoFaceDatabaseImages/CFD-WF-038-021-N.jpg"
img_str = "OurPhotos/DSC06481.JPG"
facial_features_and_values(img_str)











    


