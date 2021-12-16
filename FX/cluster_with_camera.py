import cv2
import numpy as np
from line_max import y_max
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans
from img_functions import get_c, get_half_angle, get_line_ab,get_2theta,img_process


cap = cv2.VideoCapture(0) # 0 means /dev/video0, 1 for /dev/video1, ...
#fig = plt.figure()
#ax1 = plt.axes(projection='3d')

while True :
    _, img = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))
    img_process(img)
    
    if cv2.waitKey(1)== ord("q"):
        break

#plt.show()
cap.release()
cv2.destroyAllWindows()
