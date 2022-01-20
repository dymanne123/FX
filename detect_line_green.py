import cv2
import sys
import numpy as np


cap = cv2.VideoCapture(0) # 0 is the id of your video device.
# Check if camera opened successfully
if not cap.isOpened():
    print("Unable to read camera feed")
    sys.exit(0)
    
fld = cv2.ximgproc.createFastLineDetector()
    
while(True):
    ret, img = cap.read()
 # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_green = np.array([35,43,46],dtype=np.uint8)
    upper_green = np.array([77,255,255],dtype=np.uint8)

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Bitwise-AND mask and original image
    img_mask = cv2.bitwise_and(img,img, mask= mask)
    img_mask = cv2.cvtColor(img_mask,cv2.COLOR_BGR2GRAY)
    lines = fld.detect(img_mask)
    #linesgray=cv2.cvtCOLOR(lines,cv2.COLOR_BGR2GRAY)
    lines=fd2format_s(lines)
    result_img=fld.drawSegments(img,lines)
    cv2.imshow('lines',result_img)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break




