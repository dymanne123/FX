import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, OPTICS
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

#For w=(a,b)=(cos(theta),sin(theta)), get new_w=(cos(2*theta),sin(2*theta))=(a',b')
def get_new_w(w):
    theta=math.acos(w[0])
    new_w=np.array([math.cos(2*theta),math.sin(2*theta)])
    return new_w

#For w=(a',b')=(cos(2*theta),sin(2*theta)), get orig_w=(cos(theta),sin(theta))
def get_orig_w(w):
    theta=math.acos(w[0])
    if w[1]<0:
        theta=2*math.pi-theta
    orig_w=np.array([math.cos(theta/2),math.sin(theta/2)])    
    return orig_w

#Draw a line: a*x+b*y+c=0, with c, w=(a,b) on img
def draw_lines(img,w,c):
    hei,wid,_=img.shape
    a,b=w
    n=0
    arr_to_draw=np.zeros((2,2),dtype=float) #Holds the coordinates of two points used to draw a line

    #arr_p0 stores 4 endpoints of img, arr_p stores the value of endpoints computed with the line function
    arr_p0=np.array([(0,0),(wid,0),(0,hei),(wid,hei)])
    arr_p=np.array([a*arr[0]+b*arr[1]+c for arr in arr_p0])
    #print(arr_p)
    arr_p.flatten()
    #print(arr_p)

    #check if there are endpoints on the line. If yes, use that point to draw the line.
    for i in range(4):
        if arr_p[i]==0:
           arr_to_draw[n]=arr_p0[i]
           n=n+1 
    
    #arr_l is a array to check if there exists a edge of img, whose two end points have different sign
    #If yes, one of the points for drawing the line is on this edge. 
    arr_l=np.array([arr_p[0]*arr_p[1],arr_p[0]*arr_p[2],arr_p[1]*arr_p[3],arr_p[2]*arr_p[3]])
    if arr_l[0]<0:
        arr_to_draw[n][0],arr_to_draw[n][1]=-c/a,0  #At this time, we don't need to consider if a==0.
        n=n+1
    if arr_l[1]<0:
        arr_to_draw[n][0],arr_to_draw[n][1]= 0,-c/b
        n=n+1
    if arr_l[2]<0:
        arr_to_draw[n][0],arr_to_draw[n][1]=wid,(-a*wid-c)/b
        n=n+1
    if arr_l[3]<0:
        arr_to_draw[n][0],arr_to_draw[n][1]=(-b*hei-c)/a,hei
        n=n+1
    arr_to_draw=arr_to_draw.astype('int')
    #print(arr_to_draw)
    cv2.line(img,(arr_to_draw[0][0],arr_to_draw[0][1]),(arr_to_draw[1][0],arr_to_draw[1][1]),(255,0,0),1)
    return
#display the clusters of a 2D matrix in a subplot
def display_matrix(matrix,i,j,axs,str=" ",color="red"):
    axs[i][j].scatter(matrix[:,0],matrix[:,1],c=color)
    if str!=" ":
        axs[i][j].set_title(str)
    axs[i][j].set_xlim(-1,1)
    axs[i][j].set_ylim(-1,1)

#display the clusters of an array in a subplot
def display_arr(arr,axs,str,arr_pred,i=0,j=0):
    axs[i][j].scatter(arr,np.zeros_like(arr),c=arr_pred)
    axs[i][j].set_title(str)


def pre_process_data(data):
    model=OPTICS(eps=0.8, min_samples=2)
    data_fit=model.fit(data)
    labels=data_fit.labels_
    data_processed=data[labels!=-1]
    return data_processed

def img_process(img,axs,pause_time=0.01,display_mode=False):

    filter_size = int(img.shape[1] * .05)
    if filter_size % 2 == 0:
        filter_size += 1

    """
    lower_green = np.array([35, 43, 46],dtype=np.uint8)
    upper_green = np.array([77,255,255],dtype=np.uint8)

    # Threshold the HSV image to get only blue colors
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_by_color = cv2.inRange(img_hsv, lower_green, upper_green) # HFB : Good
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
   
    # HFB :
    img_mask = cv2.bitwise_and(img, img, mask=mask_by_color)
    img_mask = cv2.cvtColor(img_mask,cv2.COLOR_BGR2GRAY)
    """

    
    red   = img[...,2]
    green = img[...,1]
    blue  = img[...,0]


    purple = np.maximum(red, blue)
    mask = purple*1.1 < green
    img_mask = np.zeros(img.shape[0:-1], dtype=np.uint8)
    img_mask[mask] = 255
    print(img_mask)

    cv2.imshow("mask", img_mask)

    
    img_blurred = cv2.GaussianBlur(img_mask, (filter_size, filter_size), 0)
    binary = np.zeros_like(img_blurred)
    binary[img_blurred > 50] = 255
    cv2.imshow("bin", binary)
    
    fld_detector = cv2.ximgproc.createFastLineDetector()
    fld_segments = fld_detector.detect(img_blurred)
    
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
        matrix_ab=np.array([get_new_w(np.array([a,b])) for a,b in matrix_orig])
        print(matrix_ab)
        if fld_segments.shape[0]>1:
            matrix_ab=pre_process_data(matrix_ab)
            
            """kmeans = KMeans(n_clusters=1).fit(matrix_ab)
            #print(kmeans.cluster_centers_)
            w= kmeans.cluster_centers_[0]"""

            w = np.average(matrix_ab, axis=0)
            
            w=w/np.linalg.norm(w)
            orig_w=get_orig_w(w) #make a2,b2 return to a,b, after clustering;
            #then get c
            distribution_c=np.array([[get_c(x1,y1,orig_w),get_c(x2,y2,orig_w)] for x1,y1,x2,y2 in fld_segments])
            distribution_c=distribution_c.reshape(-1,1)
            distribution_c=pre_process_data(distribution_c) 
            #print(distribution_c)
            kmeans_c=KMeans(n_clusters=2,random_state=0).fit(distribution_c)
            c_pred=KMeans(n_clusters=2).fit_predict(distribution_c)
            #print(kmeans_c)
            c1,c2=kmeans_c.cluster_centers_
            #print(c1,c2)
            for axs1 in axs:
                for axs2 in axs1: 
                    axs2.cla()
        
            draw_lines(img,orig_w,c1)
            draw_lines(img,orig_w,c2) 
            if (display_mode):
            #display the clusters and center points
                display_matrix(matrix_orig,0,0,axs,"original a,b","red")
                display_matrix(matrix_ab,0,1,axs,"a,b for clustering","red")
                display_matrix(np.array([orig_w]),0,0,axs,color="blue")
                display_matrix(np.array([w]),0,1,axs,color="blue")
                display_arr(distribution_c,axs,"c",c_pred,1,0) 
                axs[1][0].scatter(np.array([c1,c2]),np.zeros(2),c="red")
                plt.show()
                plt.pause(pause_time)         
    cv2.imshow("img", img)
    return   
