import cv2


cap = cv2.VideoCapture(0) # 0 means /dev/video0, 1 for /dev/video1, ...
while True :
    _, img = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fld_detector = cv2.ximgproc.createFastLineDetector()
    fld_segments = fld_detector.detect(img_gray)
    fld_segments2format = None
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