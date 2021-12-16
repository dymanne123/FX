import cv2
import numpy as np
from line_max import y_max
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans
import math
#get (a,b) knowing two points:
def get_line_ab(x1,y1,x2,y2):
    if (x2-x1!=0):
        a=(y1-y2)/(x2-x1)
        b=1
        k=math.sqrt(1+a*a)
        a=a/k
        b=b/k
    else:
        a=1
        b=0
    return (a,b)

#get c knowing the point and (a,b):
def get_c(x1,y1,x2,y2,a,b):
    c1=-a*x1-b*y1
    c2=-a*x2-b*y2
    return (c1,c2)

#we know a=cos(theta),b=sin(theta),get (a2,b2) when 2*theta:
def get_2theta(a,b):
    b2=2*a*b
    a2=a*a-b*b
    return (a2,b2)

#get (half_angle_a,half_angle_b) when theta/2:
def get_half_angle(a,b):
    half_angle_b=math.sqrt((1-a)/2)
    if b>=0:
        half_angle_a=math.sqrt((1+a)/2)
    else:
        half_angle_a=-math.sqrt((1+a)/2)
    return (half_angle_a,half_angle_b)


def draw_lines(img,w,h,a,b,c):
    p1,p2,p3,p4=-c/b,(-a*w-c)/b,-c/a,(-b*h-c)/a
    if (p1>=0) and (p1<=h) and (p2>=0) and (p2<=h):
        cv2.line(img,(0,int(p1)),(w,int(p2)),(255,0,0),1)
        return
    if (p1>=0) and (p1<=h) and (p3>=0) and (p3<=w):
        cv2.line(img,(0,int(p1)),(int(p3),0),(255,0,0),1)
        return
    if (p1>=0) and (p1<=h) and (p4>=0) and (p4<=w):
        cv2.line(img,(0,int(p1)),(int(p4),h),(255,0,0),1)
        return
    if (p2>=0) and (p2<=h) and (p3>=0) and (p3<=w):
        cv2.line(img,(w,int(p2)),(int(p3),0),(255,0,0),1)
        return
    if (p2>=0) and (p2<=h) and (p4>=0) and (p4<=h):
        cv2.line(img,(w,int(p2)),(int(p4),h),(255,0,0),1)
        return
    if (p3>=0) and (p3<=w) and (p4>=0) and (p4<=h):
        cv2.line(img,(int(p3),0),(int(p4),h),(255,0,0),1)
        return

def img_process(img):
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
    if fld_segments is not None:
        fld_segments = fld_segments.reshape((fld_segments.shape[0], 4)) # fld_segments was (n, 1, 4) shaped
    
    img_lines=fld_detector.drawSegments(img,fld_segments)
    cv2.imshow('lines',img_lines)
    if fld_segments is None:
        pass 
    else:
        matrix_ab=np.zeros((fld_segments.shape[0], 2), dtype=float)

        #knowing the start point and end point, get a2,b2 of detected lines:
        matrix_ab=np.array([get_2theta(get_line_ab(x1, y1, x2, y2)[0],get_line_ab(x1, y1, x2, y2)[1]) for x1, y1, x2, y2 in fld_segments])
        #print(matrix_ab)
        if fld_segments.shape[0]>1:
            kmeans = KMeans(n_clusters=1).fit(matrix_ab)
            print(kmeans.cluster_centers_)
            a, b= kmeans.cluster_centers_[0]
            a,b=get_half_angle(a,b)
            #make a2,b2 return to a,b, after clustering;
            #then get c
            distribution_c=np.array([get_c(x1,y1,x2,y2,a,b) for x1,y1,x2,y2 in fld_segments])
           
            distribution_c=distribution_c.reshape((distribution_c.shape[0]*2,-1))
            #print(distribution_c)
            kmeans_c=KMeans(n_clusters=2,random_state=0).fit(distribution_c)
            c1,c2=kmeans_c.cluster_centers_
            print(c1,c2)


            draw_lines(img,img.shape[0],img.shape[1],a,b,c1)
            draw_lines(img,img.shape[0],img.shape[1],a,b,c2)           
    cv2.imshow("img", img)
    return   
