import cv2
import numpy as np
from line_max import y_max
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans
from img_functions import img_process


cap = cv2.VideoCapture(0) # 0 means /dev/video0, 1 for /dev/video1, ...
plt.ion()
fig,axs=plt.subplots(2,2,figsize=(12,12))

while True :
    _, img = cap.read()
    pause_time=0.01
    img_process(img,axs,pause_time,display_mode=False)
    
    if cv2.waitKey(1)== ord("q"):
        break

#plt.show()
cap.release()
cv2.destroyAllWindows()
