import cv2
import numpy as np

def fd2format_s(lsd_output):
    lsd_reshape = lsd_output.reshape(lsd_output.shape[0], lsd_output.shape[-1])
    final_selection = np.zeros( ( lsd_output.shape[0], 8 ) )
    for x in range(lsd_output.shape[0]):
        x1,y1,x2,y2 = lsd_reshape[x].astype(int)
        final_selection[x][0] = x1
        final_selection[x][1] = y1
        final_selection[x][2] = x2
        final_selection[x][3] = y2
        A1=np.transpose( np.array([x1,y1]))
        A2 =np.transpose (np.array([x2,y2]))
        u = (A2- A1)  /np.linalg.norm(A2 - A1)
        w = - np.array([[0, 1], [-1, 0]]).dot(u)
        c = - w.dot(A1) 
        final_selection[x][4] = w[0]
        final_selection[x][5] = w[-1]
        final_selection[x][6] = c
        final_selection[x][7] = np.linalg.norm(A2 - A1)
    return final_selection
    