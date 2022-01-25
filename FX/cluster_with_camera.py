import cv2
import numpy as np
from line_max import y_max
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans
from img_functions import img_process
import time


cap = cv2.VideoCapture(0) # 0 means /dev/video0, 1 for /dev/video1, ...
plt.ion()
fig,axs=plt.subplots(2,2,figsize=(12,12))
start=time.process_time()
t_prio=0
wx_prio=0
wy_prio=0
c_prio=0
while True :
    _, img = cap.read()
    pause_time=0.01
    wx,wy,c=img_process(img,axs,pause_time,display_mode=False)
    t=time.process_time()-start
    print(wx,wy,t)
    axs[0][0].scatter(t,wx,c="red")
    axs[0][0].scatter(t,wy,c="blue")
    axs[0][1].scatter(t,c,c="blue")
    axs[1][0].scatter(t,(wx-wx_prio)/(t-t_prio),c="red")
    axs[1][0].scatter(t,(wy-wy_prio)/(t-t_prio),c="blue")
    axs[1][1].scatter(t,(c-c_prio)/(t-t_prio),c="blue")
    plt.show()
    plt.pause(0.01)
    t_prio,wx_prio,wy_prio,c_prio=t,wx,wy,c
    if cv2.waitKey(1)== ord("q"):
        break

#plt.show()
cap.release()
cv2.destroyAllWindows()
