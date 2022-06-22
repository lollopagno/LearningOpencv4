import cv2
import cv2 as cv
import numpy as np
from scipy import ndimage


def hpf3x3(img):
    """
    High pass filter with kernel 3 x 3.
    :param img:
    :return:
    """

    kernel_3x3 = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])

    k3 = ndimage.convolve(img, kernel_3x3)
    return k3


def hpf5x5(img):
    """
    High pass filter with kernel 5 x 5.
    :param img:
    :return:
    """

    kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                           [-1, 1, 2, 1, -1],
                           [-1, 2, 4, 2, -1],
                           [-1, 1, 2, 1, -1],
                           [-1, -1, -1, -1, -1]])

    k5 = ndimage.convolve(img, kernel_5x5)
    return k5


def hpf(img):
    """
    Higth pass filter on the image passed as input.
    """

    blurred = cv.GaussianBlur(img, ksize=(17, 17), sigmaX=0)  # Gaussian blur: LPF
    g_hpf = img - blurred

    return g_hpf


if __name__ == "__main__":
    img = cv2.imread("../Imgs/statue.jpg", 0)
    img = cv2.resize(img, (500, 500))

    k3 = hpf3x3(img)
    k5 = hpf5x5(img)
    hpf = hpf(img)

    cv.imshow("Original", img)
    cv.imshow("Hpf", hpf)
    cv.imshow("K3", k3)
    cv.imshow("K5", k5)

    cv.waitKey()
    cv.destroyAllWindows()
