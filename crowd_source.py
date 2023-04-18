import glob
import cv2


#press d for dark
#press l for light
#for each image always press your choice for hair first, then eyes
def survey(name):
    crowd = open(name + '_crowd.txt', 'w')
    crowd.write("hair eyes img\n")
    for image in glob.glob("./ChicagoFaceDatabaseImages/*.jpg"):
        img = cv2.imread(image)
        img = cv2.resize(img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        cv2.imshow('display', img)
        #hair first
        key1 = cv2.waitKey(0)
        cv2.imshow('display', img)
        #eyes second
        key2 = cv2.waitKey(0)
        #dark=100 light=108
        crowd.write("{:d} {:d} {:s}\n".format(key1,key2,image))
    crowd.close()