import cv2
import numpy as np
from scipy.spatial import distance as dist
def y_max(lines):
    yMax=0
    yMin=lines[0][0][1]
    for L in lines:
        
        bx,by,ex,ey = L[0]
        
        if by > yMax:
            yMax = by
        if by<yMin:
            yMin=by
    return (yMax,yMin)
def line_max(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #Create default Fast Line Detector (FSD)
    fld = cv2.ximgproc.createFastLineDetector()
    
    #Detect lines in the image
    lines = fld.detect(gray)
    
    dMax = 0
    bx_Max = 0
    by_Max = 0
    ex_Max = 0
    ey_Max = 0
    
    for L in lines:
    
        bx,by,ex,ey = L[0]
        
        # compute the Euclidean distance between the two points,
        D = dist.euclidean((bx, by), (ex, ey))
        
        if D > dMax:
            dMax = D
            bx_Max = bx
            by_Max = by
            ex_Max = ex
            ey_Max = ey
            
    lineMax = np.array([[[bx_Max, by_Max, ex_Max,ey_Max]]])
    #Draw detected lines in the image
    drawn_img = fld.drawSegments(gray,lineMax,True)
    cv2.circle(drawn_img, (bx_Max, by_Max), 1, (255,0,0), 2)#line begin
    cv2.circle(drawn_img, (ex_Max, ey_Max), 1, (0,255,0), 2)#line end
    
    cv2.imshow("FLD", drawn_img)
    cv2.waitKey(0)