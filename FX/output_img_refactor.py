"""
This modules refactors the output of the cv2 lsd detector class into linear equations
so we can manipulate the data they provide
"""
import numpy as np


def segments2canonical(cv2_segment_output):
    """
    Transforms the output of cv2 segments to an () array
    and maps to each segment the generate_coeff function

    Parameters
    -----------
    np array (n,4)
        with the coordinates of the n segment.

    Returns
    -------
    np array(n, 4)
        with (a,b,c, theta)
    """
    segment_reshape = cv2_segment_output.reshape(
        cv2_segment_output.shape[0], cv2_segment_output.shape[-1]
    )
    return np.array(map(lambda x: generate_coeff(*x), segment_reshape))


def generate_coeff(x1, y1, x2, y2):
    """
    Receives the coordinates of two points and gives the canonical
    equation of the line and the angle of the normal vector.

    Parameters
    -----------
    x1: int
        absciss of the first point
    y1: int
        ordinate of the first point
    x2: int
        absciss of the second point
    y2: int
        ordinate of the second point

    Returns
    -------
    np.array(4)
        with (a,b,c,theta)
    """
    dir_vec = np.array([x1 - x2, y1 - y2])
    dir_vec_n = dir_vec / np.sqrt(np.sum(dir_vec ** 2))
    a, b = -dir_vec_n[-1], dir_vec_n[0]
    c = -(x1 * a + y1 * b)
    theta = a * b
    return np.array([a, b, c, theta])


def fd2format_s(lsd_output):
    lsd_reshape = lsd_output.reshape(lsd_output.shape[0], lsd_output.shape[-1])
    final_selection = np.zeros((lsd_output.shape[0], 8))
    for x in range(lsd_output.shape[0]):
        x1, y1, x2, y2 = lsd_reshape[x].astype(int)
        final_selection[x][0] = x1
        final_selection[x][1] = y1
        final_selection[x][2] = x2
        final_selection[x][3] = y2
        A1 = np.transpose(np.array([x1, y1]))
        A2 = np.transpose(np.array([x2, y2]))
        u = (A2 - A1) / np.linalg.norm(A2 - A1)
        w = -np.array([[0, 1], [-1, 0]]).dot(u)
        c = -w.dot(A1)
        final_selection[x][4] = w[0]
        final_selection[x][5] = w[-1]
        final_selection[x][6] = c
        final_selection[x][7] = np.linalg.norm(A2 - A1)
    return final_selection


def tracage_courbe(img, canonical_segments):
    h, w, c = img.shape
    points = np.array([[0, 0], [w, 0], [h, w], [0, h]])


def mapping(points, a, b, c):
    values = np.array(map(lambda x: x[0] * a + x[1] * b + c, points))
    signs = np.sign(values)
    if signs[0] != signs[2] and signs[1] != signs[3]:
        if signs[1] == signs[2]:
            # vertical
            pass
        # horizontal
    elif signs[0] != signs[2]:
        if signs[2] > signs[0]:
            # |_
            pass
        else:
            # -|
            pass
    elif signs[1] != signs[3]:
        if signs[1] > signs[3]:

            # |-
            pass
        else:
            # _|
            pass
