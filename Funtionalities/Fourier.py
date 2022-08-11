import cv2 as cv
import numpy as np
import matplotlib.pyplot as plot


def fastFourierTransform(img):
    """
    Calculate the magnitude spectrum, fft and ifft.
    """

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fft = np.log(np.abs(fshift))

    row, cols = img.shape
    crow, ccol = row // 2, cols // 2
    fshift[crow - 30: crow + 30, ccol - 30: ccol + 30] = 0

    f_ishift = np.fft.ifftshift(fshift)
    ifft = np.fft.ifft2(f_ishift)
    ifft = np.abs(ifft)

    plot.subplot(221), plot.imshow(fft, cmap="gray")
    plot.title("FFT"), plot.xticks([]), plot.yticks([])

    plot.subplot(222), plot.imshow(ifft, cmap="gray")
    plot.title("IFFT"), plot.xticks([]), plot.yticks([])
    plot.show()


img = cv.imread("../Imgs/statue.jpg", 0)
img = cv.resize(img, (500, 500))
fastFourierTransform(img)
