import cv2
import numpy as np
from line_max import y_max
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans
import math
#For a line: a*x+b*y+c=0, get w=(a,b) with two points(x1,y1), (x2,y2)
def get_line_ab(x1,y1,x2,y2):
    A1=np.transpose(np.array([x1,y1]))
    A2 =np.transpose (np.array([x2,y2]))
    u = (A2- A1)  /np.linalg.norm(A2 - A1)
    w = - np.array([[0, 1], [-1, 0]]).dot(u)
    if w[1]<0:
        w=-w
    return w

#For a line: a*x+b*y+c=0, get c knowing a point(x,y) on it and w=(a,b):
def get_c(x,y,w):
    c=- w.dot(np.transpose(np.array([x,y]))) 
    return c

#For w=(a,b)=(cos(theta),sin(theta)), get a new_w=(cos(2*theta),sin(2*theta))
def get_new_w(w):
    theta=math.acos(w[0])
    new_w=np.array([math.cos(2*theta),math.sin(2*theta)])
    return new_w

#For w=(cos(2*theta),sin(2*theta)), get orig_w=(cos(theta),sin(theta))
def get_orig_w(w):
    theta=math.acos(w[0])
    if w[1]<0:
        theta=2*math.pi-theta
    orig_w=np.array([math.cos(theta/2),math.sin(theta/2)])    
    return orig_w

#Draw a line: a*x+b*y+c=0, with w=(a,b),c on img
def draw_lines(img,w,c):
    hei,wid,num=img.shape
    a,b=w
    n=0
    arr_to_draw=np.zeros((2,2))
    arr_p0=np.array([(0,0),(wid,0),(0,hei),(wid,hei)])
    arr_p=np.array([a*arr[0]+b*arr[1]+c for arr in arr_p0])
    #print(arr_p)
    arr_p.flatten()
    #print(arr_p)
    for i in range(4):
        if arr_p[i]==0:
           arr_to_draw[n]=arr_p0[i]
           n=n+1 
    arr_l=np.array([arr_p[0]*arr_p[1],arr_p[0]*arr_p[2],arr_p[1]*arr_p[3],arr_p[2]*arr_p[3]])
    if arr_l[0]<0:
        arr_to_draw[n]=np.array([-c/a,0])
        n=n+1
    if arr_l[1]<0:
        arr_to_draw[n]=np.array([0,-c/b])
        n=n+1
    if arr_l[2]<0:
        arr_to_draw[n]=np.array([wid,(-a*wid-c)/b])
        n=n+1
    if arr_l[3]<0:
        arr_to_draw[n]=np.array([(-b*hei-c)/a,hei])
        n=n+1
    arr_to_draw=arr_to_draw.astype('int')
    #print(arr_to_draw)
    cv2.line(img,(arr_to_draw[0][0],arr_to_draw[0][1]),(arr_to_draw[1][0],arr_to_draw[1][1]),(255,0,0),1)
    return

def display_matrix(matrix_ab,matrix_ab_orig,arr_c,axs):
    plt.cla()
    axs[0].scatter(matrix_ab[:,0],matrix_ab[:,1])
    axs[1].scatter(matrix_ab_orig[:,0],matrix_ab_orig[:,1])
    axs[2].scatter(arr_c,np.zeros_like(arr_c))
    plt.show()
    plt.pause(100)

def img_process(img,axs):
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
        matrix_orig=np.zeros((fld_segments.shape[0], 2), dtype=float)
        #knowing the start point and end point, get a2,b2 of detected lines:
        matrix_orig=np.array([get_line_ab(x1, y1, x2, y2) for x1,y1,x2,y2 in fld_segments])
        matrix_ab=np.array([get_new_w(get_line_ab(x1, y1, x2, y2)) for x1,y1,x2,y2 in fld_segments])
        print(matrix_ab)
        if fld_segments.shape[0]>1:
            kmeans = KMeans(n_clusters=1).fit(matrix_ab)
            #print(kmeans.cluster_centers_)
            w= kmeans.cluster_centers_[0]
            w=get_orig_w(w)
            #make a2,b2 return to a,b, after clustering;
            #then get c
            distribution_c=np.array([[get_c(x1,y1,w),get_c(x2,y2,w)] for x1,y1,x2,y2 in fld_segments])
            distribution_c=distribution_c.reshape(-1,1)
        
            #print(distribution_c)
            kmeans_c=KMeans(n_clusters=2,random_state=0).fit(distribution_c)
            c1,c2=kmeans_c.cluster_centers_
            #print(c1,c2)
            display_matrix(matrix_ab,matrix_orig,distribution_c,axs)

            draw_lines(img,w,c1)
            draw_lines(img,w,c2)           
    cv2.imshow("img", img)
    return   
