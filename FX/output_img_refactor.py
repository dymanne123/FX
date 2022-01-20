"""
This modules refactors the output of the cv2 lsd detector class into linear equations
so we can manipulate the data they provide
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math
import timeit


def segments2canonical(cv2_segment_output):
    """
    Transforms the output of cv2 segments to an  array
    and maps to each segment the generate_coeff function
    It uses generate_coeff_sum (math package)

    Parameters
    -----------
    np array (n,4)
        with the coordinates of the n segment.

    Returns
    -------
    np array(n, 4)
        with (a,b,c, theta)
    """
    f = lambda x: generate_coeff_sum(x)
    return np.apply_along_axis(f, 1, cv2_segment_output)


def segments2canonical_arr(cv2_segment_output):
    """
    Transforms the output of cv2 segments to an  array
    and maps to each segment the generate_coeff function
    It uses generate_coeff_arr (math package)

    Parameters
    -----------
    np array (n,4)
        with the coordinates of the n segment.

    Returns
    -------
    np array(n, 4)
        with (a,b,c, theta)
    """
    f = lambda x: generate_coeff_arr(x)
    return np.apply_along_axis(f, 1, cv2_segment_output)


def generate_coeff_arr(array):
    """
    Receives the array containing the coordinates of two points and gives the canonical
    equation of the line and the angle of the normal vector.
    It's equivalent to generate_coeff_sum but here we use the numpy package

    Parameters
    -----------
    array: np.array(4)
        with the coordinates of the segments

    Returns
    -------
    np.array(4)
        with (a,b,c,theta)
    """
    dir_vec = np.array([array[0] - array[2], array[1] - array[3]])
    dir_vec_n = dir_vec / np.sqrt(np.sum(dir_vec ** 2))
    a, b = -dir_vec_n[-1], dir_vec_n[0]
    if b < 0:
        a, b = -a, -b
    c = -(array[0] * a + array[1] * b)
    theta = math.acos(a)
    return np.array([a, b, theta, c])


def generate_coeff_sum(array):
    """
    Receives the array containing the coordinates of two points and gives the canonical
    equation of the line and the angle of the normal vector.
    It's equivalent to generate_coeff but we use the math package

    Parameters
    -----------
    array: np.array(4)
        with the coordinates of the segments

    Returns
    -------
    np.array(4)
        with (a,b,theta,c)
    """
    a1, b1 = array[2] - array[0], array[3] - array[1]
    b, a = a1 / math.sqrt(a1 ** 2 + b1 ** 2), -b1 / math.sqrt(a1 ** 2 + b1 ** 2)
    if b < 0:
        a, b = -a, -b
    theta = math.acos(a)
    c = -(array[0] * a + array[1] * b)
    return np.array([a, b, theta, c], dtype="f")


def change2theta_c(arrabtheta, arrc):
    """
    We compute the two best lines that describes all equation using Kmeans

    Parameters
    -----------
    arrabtheta: np.array(n,3)
        array containing the coefficients (a,b,theta) for each line and n is the
        number of lines.

    arrc: np.array(n,1)
        array containing all the c coefficient for each line

    Returns
    --------
    np.array(2,3)
        2d numpy array containing (a,b,c) that are the

    """
    f = lambda x: np.array([math.cos(2 * x[-1]), math.sin(2 * x[-1])])
    matrix = np.apply_along_axis(f, 1, arrabtheta)
    kmeans_ab = KMeans(n_clusters=1).fit(matrix)
    print(matrix)
    ab = kmeans_ab.cluster_centers_
    a_prime, b_prime = ab.flatten()
    print(ab)
    plt.scatter(matrix[:, 0], matrix[:, 1])
    plt.scatter(a_prime, b_prime, s=80)
    plt.xlabel("valeur de a")
    plt.ylabel("valeur de b")
    plt.title("kmeans de la tranformation de a et b")
    plt.show()
    theta_prime = math.acos(a_prime)
    if b_prime < 0:
        theta_prime = 2 * math.pi - theta_prime
    a, b = math.cos(theta_prime / 2), math.sin(theta_prime / 2)
    arrc = arrc.reshape(-1, 1)
    kmeancs_c = KMeans(n_clusters=2, random_state=0).fit(arrc)
    clusters_c = kmeancs_c.cluster_centers_
    c1, c2 = clusters_c.flatten()
    return np.array([[a, b, c1], [a, b, c2]])


def draw_lines(img, lines):
    """
    TO DO
    Parameters
    ----------
    img: image

    lines: np.array(2,3)

    Returns
    --------
    image with the two lines that are drawn
    """
    h, w, _ = img.shape
    a1, b1, c1 = lines[0]
    a2, b2, c2 = lines[1]
    arr_p0 = np.array([(0, 0), (w, 0), (0, h), (w, h)])
    arr_p1 = np.array([a1 * arr[0] + b1 * arr[1] + c1 for arr in arr_p0])
    arr_p2 = np.array([a2 * arr[0] + b2 * arr[1] + c2 for arr in arr_p0])


def img_procesings(img):
    """
    The filtring proecessing used to detect the green lightsaber from
    an image

    Parameters
    -----------
    img:
        An image where we want to detect the green lightsaber

    Output
    -------
    img:
         The same image received in parameters with two parallel lines
         that represent the position of the saber in the image.
    """
    lower_green = np.array([35, 43, 46], dtype=np.uint8)
    upper_green = np.array([77, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only blue colors
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_by_color = cv2.inRange(img_hsv, lower_green, upper_green)  # HFB : Good

    # HFB :
    img_gray_blurred = cv2.GaussianBlur(mask_by_color, (31, 31), 0)
    img_gray_blurred[img_gray_blurred > 100] = 255
    img_gray_blurred[img_gray_blurred < 100] = 0
    # cv2.imshow("gree-detect", img_gray_blurred)

    fld_detector = cv2.ximgproc.createFastLineDetector()
    fld_segments = fld_detector.detect(img_gray_blurred)
    if fld_segments is not None:
        fld_segments = fld_segments.reshape(
            (fld_segments.shape[0], 4)
        )  # fld_segments was (n, 1, 4) shaped

    img_lines = fld_detector.drawSegments(img, fld_segments)
    cv2.imshow("lines", img_lines)

    if fld_segments is None:
        pass
    else:
        canonical_eq = segments2canonical(fld_segments)
        # research abt matrix
        abtheta = canonical_eq[:, :3]
        c_values = canonical_eq[:, -1]
        result = change2theta_c(abtheta, c_values)
        # plt.figure()
        # plt.scatter(abtheta[0], abtheta[1])
        # plt.xlabel("valeur de a")
        # plt.ylabel("valeur de b")
        # plt.title("transformations")


if __name__ == "__main__":

    test_arr = np.array([[-1, 1, -3, 4], [0, -8, 1, -7], [3, -0.5, 10, -0.5]])
    test_arrs = np.array([[-1, 1, -3, 4]])
    start = timeit.timeit()
    test_can = segments2canonical_arr(test_arr)
    print(test_can[:, -1], test_can[:, :3])
    end = timeit.timeit()
    test_can2 = segments2canonical(test_arr)
    awa, uwu = test_can2[:, -1], test_can2[:, :3]
    liness = change2theta_c(uwu, awa)
    print(liness)
