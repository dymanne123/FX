import cv2
from lines import fd2format_s
import numpy as np

cap = cv2.VideoCapture(0) # 0 means /dev/video0, 1 for /dev/video1, ...
while True :
    _, img = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    lower_green = np.array([35,43,46],dtype=np.uint8)
    upper_green = np.array([77,255,255],dtype=np.uint8)

    # Threshold the HSV image to get only blue colors
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_by_color = cv2.inRange(img_hsv, lower_green, upper_green)
    # Bitwise-AND mask and original image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result_hsv_filter = cv2.bitwise_and(img,img, mask = mask_by_color)
    result_hsv2bgr = cv2.cvtColor(result_hsv_filter, cv2.COLOR_HSV2BGR)
    result_hsv2gray = cv2.cvtColor(result_hsv2bgr, cv2.COLOR_BGR2GRAY)


    # img_gray_blurred = cv2.GaussianBlur(img_gray, (5,5),0)
    img_gray_blurred = cv2.GaussianBlur(result_hsv2gray, (5,5),0)
    fld_detector = cv2.ximgproc.createFastLineDetector()
    fld_segments = fld_detector.detect(img_gray_blurred)
    #fld_segments2format = fd2format_s(fld_segments)


    #pherhaps we can filter by longitude

    out_fld = fld_detector.drawSegments(img, fld_segments)

    cv2.imshow("img",out_fld)

    #1 mettre l'image en gris et faire lsd
    # appliquer le out a lsd2format
    # drawlines apply
    if cv2.waitKey(1)== ord("q"):
        break

    
cap.release()
cv2.destroyAllWindows()