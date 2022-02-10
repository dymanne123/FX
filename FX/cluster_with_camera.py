import cv2
import numpy as np
from line_max import y_max
from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans
from img_functions import img_process
import time
from Saber_Sound import SaberSound

num=10
def get_prio(arr,index):
    if (index!=0):
        return arr[index-1]
    else:
        return arr[num-1]

cap = cv2.VideoCapture(0) # 0 means /dev/video0, 1 for /dev/video1, ...
plt.ion()
fig,axs=plt.subplots(2,2,figsize=(12,12))
start=time.process_time()
saber_sound=SaberSound()
saber_sound.start()

#4 arrays to store 10 latest values of wx,wy,c and t
wx_arr=np.zeros(num)
wy_arr=np.zeros(num)
c_arr=np.zeros(num)
t_arr=np.zeros(num)
index=0

while True :
    _, img = cap.read()
    pause_time=0.01
    #get line parameters with img_functions
    wx,wy,c=img_process(img,axs,pause_time,display_mode=False)
    if (wx,wy,c!=0,0,0):
        wx=wx*0.1+get_prio(wx_arr,index)*0.9
        wy=wy*0.1+get_prio(wy_arr,index)*0.9
        c=c*0.1+get_prio(c_arr,index)*0.9
        wx_arr[index],wy_arr[index],c_arr[index]=wx,wy,c
        t=time.process_time()
        t_arr[index]=t
        print("t:",t)

        if (False):
            axs[0][0].scatter(t,wx,c="red")
            axs[0][0].scatter(t,wy,c="blue")
            axs[0][1].scatter(t,c,c="blue")
            axs[1][0].scatter(t,(wx-get_prio(wx_arr,index))/(t-get_prio(t_arr,index)),c="red")
            axs[1][0].scatter(t,(wy-get_prio(wy_arr,index))/(t-get_prio(t_arr,index)),c="blue")
            axs[1][1].scatter(t,(c-get_prio(c_arr,index))/(t-get_prio(t_arr,index)),c="blue")
            plt.show()
            plt.pause(0.01)
        
        if (index==num-1):
            angle_v=max((np.absolute(wx_arr[num-1])-np.absolute(wx_arr[0]))/(t_arr[num-1]-t_arr[0]),(np.absolute(wy_arr[num-1])-np.absolute(wy_arr[0]))/(t_arr[num-1]-t_arr[0]))
            c_v=(np.absolute(c_arr[num-1])-np.absolute(c_arr[0]))/(t_arr[num-1]-t_arr[0])
            if (angle_v)>0.05:
                saber_sound.set_value(min(1000*angle_v,100))
                time.sleep(0.1)
                print(angle_v)
            else:
                if (c_v)>15:
                    saber_sound.set_value(min(c_v*10,100))
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
