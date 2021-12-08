import cv2
import numpy as np
from line_max import y_max
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


fig = plt.figure()
ax1 = plt.axes(projection='3d')

img = cv2.imread("./green_sable.png") 
cv2.imshow('lines',img)

lower_green = np.array([35,43,46],dtype=np.uint8)
upper_green = np.array([77,255,255],dtype=np.uint8)

# Threshold the HSV image to get only blue colors
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask_by_color = cv2.inRange(img_hsv, lower_green, upper_green) # HFB : Good
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

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
    matrix_abc=[[0 for j in range(3)] for i in range(fld_segments.shape[0])]
    #knowing the start point and end point, get a,b,c of detected lines.
    for i in range(fld_segments.shape[0]):
        array_abc=get_line_abc(fld_segments[i][0][0],fld_segments[i][0][1],fld_segments[i][0][2],fld_segments[i][0][3])
        matrix_abc[i]=array_abc
        ax1.scatter3D(array_abc[0],array_abc[1],array_abc[2], cmap='Blues')
    print(matrix_abc)    
    #get center points (a,b,c) of two cluster groups
    if fld_segments.shape[0]>1:
        kmeans=KMeans(n_clusters=2, random_state=0).fit(matrix_abc)
        print(kmeans.cluster_centers_)
        a1=kmeans.cluster_centers_[0][0]
        b1=kmeans.cluster_centers_[0][1]
        c1=kmeans.cluster_centers_[0][2]
        a2=kmeans.cluster_centers_[1][0]
        b2=kmeans.cluster_centers_[1][1]
        c2=kmeans.cluster_centers_[1][2]
        x_cloud=[k[0] for k in matrix_abc]
        y_cloud=[k[1] for k in matrix_abc]
        z_cloud=[k[2] for k in matrix_abc]

        ymax,ymin=y_max(fld_segments)
        cv2.line(img,(int((-ymin-c1)/a1),int(ymin)),(int((-ymax-c1)/a1),int(ymax)),(255,0,0),1)
        cv2.line(img,(int((-ymin-c2)/a2),int(ymin)),(int((-ymax-c2)/a2),int(ymax)),(255,0,0),1)
        
        
    
cv2.imshow("img", img)
cv2.waitKey(0)
plt.show()