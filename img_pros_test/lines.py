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

def draw_lines(img, lines, color, thickness) :
    h,w,c = img.shape
    for line in lines:
        x1, y1, x2, y2, a ,b,c,n = line
        if a ==0 and b !=0:
            y3 = -c/b
            y4= y3
            x3 = 0
            x4 = h
            cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)),color, thickness )
        elif a!= 0:
            y3 = 0
            x3 = (-b * y3 - c)/(a)
            y4 = h
            x4 = (-b* y4 -c )/(a)
            if x3 > w:
                x3 = h
                y3 = (-a*x3-c)/b
                if x4<0:
                    x4 = 0
                    y4 = (-a*x3-c)/b
                elif x4 >w:
                    x4 = w
                    y4 = (-a*x3-c)/b
            elif x3 < 0:
                x3 = 0
                y3 = (-a*x3-c)/b
                if x4>w:
                    x4 = w
                    y4 = (-a*x3-c)/b 
            cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)),color, thickness )
        else:
            pass
        
def classif_l_r(lines):
    #We classify depending on the signs of a(4th pos) and b(5th pos)
    line_r_pos = lines[lines[:,4] > 0]
    line_r_pos = line_r_pos[line_r_pos[:, 5] > 0]
    line_r_neg = lines[lines[:,4] < 0]
    line_r_neg = line_r_neg[line_r_neg[:, 5] < 0]
    line_r = np.concatenate((line_r_pos,line_r_neg), axis=0)
    line_left = lines[lines[:,4] < 0]
    line_left = line_left[line_left[:,5] >0]
    line_left2= lines[lines[:,4]>0]
    line_left2= line_left2[line_left2[:,5]<0]
    line_l = np.concatenate((line_left,line_left2), axis=0)
    return (line_r,line_l)