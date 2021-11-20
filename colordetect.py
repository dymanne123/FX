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
    res = cv2.bitwise_and(img,img, mask= mask)
    res_bw = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    res_bw_fin = cv2.cvtColor(res_bw, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('video',img)
    #cv2.imshow('mask',mask)
    cv2.imshow('res',res_bw_fin)
    #press esc to end
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


