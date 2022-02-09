import cv2
import numpy as np
from line_max import x_max
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans
#To get a,b,c, knowing (x1,y1),(x2,y2)
def get_line_abc(x1,y1,x2,y2):
    if (x2-x1!=0):
        a=(y1-y2)/(x2-x1)
        b=1
        c=(x1*y2-x2*y1)/(x2-x1)
    else:
        a=1
        b=0
        c=-x1
    return (a,b,c)


cap = cv2.VideoCapture(0) # 0 means /dev/video0, 1 for /dev/video1, ...
fig = plt.figure()
ax1 = plt.axes(projection='3d')

while True :
    _, img = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    lower_green = np.array([35,43,46],dtype=np.uint8)
    upper_green = np.array([77,255,255],dtype=np.uint8)

    # Threshold the HSV image to get only blue colors
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_by_color = cv2.inRange(img_hsv, lower_green, upper_green) # HFB : Good
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    """
    # Bitwise-AND mask and original image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result_hsv_filter = cv2.bitwise_and(img,img, mask = mask_by_color)
    result_hsv2bgr = cv2.cvtColor(result_hsv_filter, cv2.COLOR_HSV2BGR)
    result_hsv2gray = cv2.cvtColor(result_hsv2bgr, cv2.COLOR_BGR2GRAY)


    # img_gray_blurred = cv2.GaussianBlur(img_gray, (5,5),0)
    img_gray_blurred = cv2.GaussianBlur(result_hsv2gray, (5,5), 0)
    fld_detector = cv2.ximgproc.createFastLineDetector()
    fld_segments = fld_detector.detect(img_gray_blurred)
    #It can be none type object
    if fld_segments is None:
        pass 
    else:
        fld_segments2format = fd2format_s(fld_segments)
    #pherhaps we can filter by longitude
    out_fld = fld_detector.drawSegments(img, fld_segments)
    """
    # HFB :
    img_gray_blurred = cv2.GaussianBlur(mask_by_color, (31, 31), 0)
    img_gray_blurred[img_gray_blurred>100] = 255
    img_gray_blurred[img_gray_blurred<100] = 0
    #cv2.imshow("gree-detect", img_gray_blurred)
    fld_detector = cv2.ximgproc.createFastLineDetector()
    fld_segments = fld_detector.detect(img_gray_blurred)
    img_lines=fld_detector.drawSegments(img,fld_segments)
    cv2.imshow('lines',img_lines)
    if fld_segments is None:
        pass 
    else:
        #print(fld_segments)
        matrix_abc=[[0 for j in range(3)] for i in range(fld_segments.shape[0])]
        #knowing the start point and end point, get a,b,c of detected lines.
        for i in range(fld_segments.shape[0]):
            array_abc=get_line_abc(fld_segments[i][0][0],fld_segments[i][0][1],fld_segments[i][0][2],fld_segments[i][0][3])
            matrix_abc[i]=array_abc
            ax1.scatter3D(array_abc[0],array_abc[1],array_abc[2], cmap='Blues')
        #fld_segments2format = fd2format_s(fld_segments)

        #get center points (a,b,c) of two cluster groups
        if fld_segments.shape[0]>1:
            kmeans = KMeans(n_clusters=2, random_state=0).fit(matrix_abc)
            #print(kmeans.cluster_centers_)
            a1=kmeans.cluster_centers_[0][0]
            b1=kmeans.cluster_centers_[0][1]
            c1=kmeans.cluster_centers_[0][2]
            a2=kmeans.cluster_centers_[1][0]
            b2=kmeans.cluster_centers_[1][1]
            c2=kmeans.cluster_centers_[1][2]

            xmax,xmin=x_max(fld_segments)
            lines_todraw=np.zeros((2,1,4))
            """
            lines_todraw[0,0,0]=xmin
            lines_todraw[0,0,1]=-a1*xmin-c1
            lines_todraw[0,0,2]=xmax
            lines_todraw[0,0,3]=-a1*xmax-c1
            lines_todraw[1,0,0]=xmin
            lines_todraw[1,0,1]=-a2*xmin-c2
            lines_todraw[1,0,2]=xmax
            lines_todraw[1,0,3]=-a2*xmin-c2
            """
            cv2.line(img,(int(xmin),int(-a1*xmin-c1)),(int(xmax),int(-a1*xmax-c1)),(255,0,0),3,4)
            cv2.line(img,(int(xmin),int(-a2*xmin-c2)),(int(xmax),int(-a2*xmin-c2)),(255,0,0),3,4)
            #lines_todraw=[[[xmin, -a1*xmin-c1,xmax,-a1*xmax-c1 ]][[xmin,-a2*xmin-c2,xmax,-a2*xmin-c2]]]
            #fld_detector.drawSegments(img,lines_todraw,(0,255,255),1)
            #print(lines_todraw)
        
    
    
    cv2.imshow("img", img)
    #1 mettre l'image en gris et faire lsd
    # appliquer le out a lsd2format
    # drawlines apply
    if cv2.waitKey(1)== ord("q"):
        break

#plt.show()
cap.release()
cv2.destroyAllWindows()
