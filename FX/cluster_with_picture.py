import cv2
import numpy as np
from line_max import y_max
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans
from img_functions import img_process
#To get a,b,c, knowing (x1,y1),(x2,y2)



#cap = cv2.VideoCapture(0) # 0 means /dev/video0, 1 for /dev/video1, ...
plt.ion()
fig,axs=plt.subplots(2,2,figsize=(12,12))

img = cv2.imread("./green_sable.png") 
img_process(img,axs,display_mode=True)

plt.pause(100)
#cap.release()
#cv2.destroyAllWindows()
