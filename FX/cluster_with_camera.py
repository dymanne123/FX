import cv2
import numpy as np
from line_max import y_max
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans
from img_functions import img_process
import time
import SaberSound

def get_prio(arr,index):
    if (index!=0):
        return arr[index-1]
    else:
        return arr[9]

cap = cv2.VideoCapture(0) # 0 means /dev/video0, 1 for /dev/video1, ...
plt.ion()
fig,axs=plt.subplots(2,2,figsize=(12,12))
start=time.process_time()
saber_sound=SaberSound.SaberSound()
saber_sound.start()
#4 arrays to store 10 latest values of wx,wy,c and t
wx_arr=np.zeros(10)
wy_arr=np.zeros(10)
c_arr=np.zeros(10)
t_arr=np.zeros(10)
index=0
while True :
    _, img = cap.read()
    pause_time=0.01
    wx,wy,c=img_process(img,axs,pause_time,display_mode=False)
    wx_arr[index],wy_arr[index],c_arr[index]=wx,wy,c
    wx=wx*0.1+get_prio(wx_arr,index)*0.9
    wy=wy*0.1+get_prio(wy_arr,index)*0.9
    c=c*0.1+get_prio(c_arr,index)*0.9
    t=time.process_time()
    t_arr[index]=t
    #print(wx,wy,t)
    """
    axs[0][0].scatter(t,wx,c="red")
    axs[0][0].scatter(t,wy,c="blue")
    axs[0][1].scatter(t,c,c="blue")
    axs[1][0].scatter(t,(wx-wx_prio)/(t-t_prio),c="red")
    axs[1][0].scatter(t,(wy-wy_prio)/(t-t_prio),c="blue")
    axs[1][1].scatter(t,(c-c_prio)/(t-t_prio),c="blue")
    plt.show()
    plt.pause(0.01)
    """
    if (index==9):
        angle_v=max((np.absolute(wx_arr[9])-np.absolute(wx_arr[0]))/(t_arr[9]-t_arr[0]),(np.absolute(wy_arr[9])-np.absolute(wy_arr[0]))/(t_arr[9]-t_arr[0]))
        c_v=(np.absolute(c_arr[9])-np.absolute(c_arr[0]))/(t_arr[9]-t_arr[0])
        if (angle_v)>0.05:
            saber_sound.set_value(min(1000*angle_v,100))
            time.sleep(0.1)
            print(angle_v)
        else:
            if (c_v)>100:
                saber_sound.set_value(min(c_v,100))
                time.sleep(0.1)
                print("c",c_v)
        saber_sound.set_value(0)
        index=-1
    index=index+1
    if cv2.waitKey(1)== ord("q"):
        break

#plt.show()
cap.release()
cv2.destroyAllWindows()
