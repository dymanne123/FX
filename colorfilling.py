import cv2
import sys

cap = cv2.VideoCapture(0) # 0 is the id of your video device.
# Check if camera opened successfully
if not cap.isOpened():
    print("Unable to read camera feed")
    sys.exit(0)
    

    
while(True):
    ret, img = cap.read()
    if ret: 
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
        #cv2.imshow('Binary image', thresh)
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        img_copy = img.copy()
        cv2.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow('None approximation', img_copy)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
        opening = cv2.morphologyEx(img_copy, cv2.MORPH_OPEN, kernel, iterations=2)
        mask=opening
        cv2.imshow('opening', opening)
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


