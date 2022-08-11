import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def createMedianMask(disparityMap, validDepthMask, rect=None):
    """
    Return a mask selecting the median layer, plus shadows.
    """

    if rect is not None:
        x, y, w, h = rect
        disparityMap = disparityMap[y:y + h, x:x + w]
        validDepthMask = validDepthMask[y:y + h, x:x + w]

    median = np.median(disparityMap)

    return np.where((validDepthMask == 0) | (abs(disparityMap - median) < 12), 255, 0).astype(np.uint8)


def disparityMap():
    # Variables to calculate the disparity map.

    global minDisparity
    minDisparity = 16

    global numDisparities
    numDisparities = 192 - minDisparity
    blockSize = 5
    uniquenessRatio = 1
    speckleWindowSize = 3
    speckleRange = 3
    disp12MaxDiff = 200
    P1 = 600
    P2 = 2400

    global stereo
    stereo = cv.StereoSGBM_create(
        minDisparity=minDisparity,
        numDisparities=numDisparities,
        blockSize=blockSize,
        uniquenessRatio=uniquenessRatio,
        speckleRange=speckleRange,
        speckleWindowSize=speckleWindowSize,
        disp12MaxDiff=disp12MaxDiff,
        P1=P1, P2=P2)

    # imgL = cv.resize(cv.imread('../Imgs/DisparityMap/vase1.jpg'), (450, 800))
    # imgR = cv.resize(cv.imread('../Imgs/DisparityMap/vase2.jpg'), (450, 800))

    global imgL
    imgL = cv.resize(cv.imread('../Imgs/DisparityMap/video1.jpg'), (800, 450))

    global imgR
    imgR = cv.resize(cv.imread('../Imgs/DisparityMap/video2.jpg'), (800, 450))

    cv.namedWindow('Disparity')
    cv.createTrackbar('blockSize', 'Disparity', blockSize, 21, disparityMapUpdate)
    cv.createTrackbar('uniquenessRatio', 'Disparity', uniquenessRatio, 50, disparityMapUpdate)
    cv.createTrackbar('speckleWindowSize', 'Disparity', speckleWindowSize, 200, disparityMapUpdate)
    cv.createTrackbar('speckleRange', 'Disparity', speckleRange, 50, disparityMapUpdate)
    cv.createTrackbar('disp12MaxDiff', 'Disparity', disp12MaxDiff, 250, disparityMapUpdate)

    # Initialize the disparity map. Show the disparity map and images.
    disparityMapUpdate()

    # Wait for the user to press any key.
    # Meanwhile, update() will be called anytime the user moves a slider.
    cv.waitKey()


def disparityMapUpdate(sliderValue=0):
    """
    Calcute the disparity map.
    """

    try:
        stereo.setBlockSize(cv.getTrackbarPos('blockSize', 'Disparity'))
        stereo.setUniquenessRatio(cv.getTrackbarPos('uniquenessRatio', 'Disparity'))
        stereo.setSpeckleWindowSize(cv.getTrackbarPos('speckleWindowSize', 'Disparity'))
        stereo.setSpeckleRange(cv.getTrackbarPos('speckleRange', 'Disparity'))
        stereo.setDisp12MaxDiff(cv.getTrackbarPos('disp12MaxDiff', 'Disparity'))
    except:
        # Quando si crea la prima trackbar, l'update verrà chiamata e andra in eccezzione
        # perche gli altri trackbar non sono stati ancora creati.
        pass

    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    cv.imshow('Left', imgL)
    cv.imshow('Right', imgR)
    cv.imshow('Disparity', (disparity - minDisparity) / numDisparities)


def grabCut():
    """
    GrabCut alghorithm implementation.
    """

    binarizedImg = True

    original = cv.imread('../Imgs/statue_small.jpg.jpg')
    gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)

    img = original.copy()
    mask = np.zeros(img.shape[:2], np.uint8)
    mode = cv.GC_INIT_WITH_RECT

    if binarizedImg:
        binarized_image = cv.adaptiveThreshold(gray, maxValue=1, adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               thresholdType=cv.THRESH_BINARY, blockSize=9, C=7)

        mask[:] = cv.GC_PR_BGD
        mask[binarized_image == 0] = cv.GC_PR_FGD

        mode = cv.GC_INIT_WITH_MASK

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (100, 1, 421, 378)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, mode=mode)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    plt.subplot(121)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("grabcut")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(122)
    plt.imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
    plt.title("original")
    plt.xticks([])
    plt.yticks([])

    plt.show()


def watershed():
    """
    Watershed alghorithm implementation.
    https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
    """

    img = cv.imread('../Imgs/card.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    # Remove noise.
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # Find the sure background region.
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Find the sure foreground region.
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2,
                                          5)  # To obtain the derived representation of a binary image,
    # where the value of each pixel is replaced by its distance to the nearest background pixel. The L2 distance is the euclidean distance

    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # Find the unknown region.
    unknown = cv.subtract(sure_bg, sure_fg)
    cv.imshow("unknow", unknown)

    # Label the foreground objects.
    ret, markers = cv.connectedComponents(
        sure_fg)  # Returns the total number of labels [0, N-1] where 0 represents the background label.
    # Markers is an array with the same size of the image

    # Add one to all labels so that sure background is not 0, but 1.
    markers += 1

    # Label the unknown region as 0.
    markers[unknown == 255] = 0

    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

    cv.waitKey()


if __name__ == "__main__":
    # disparityMap()  # To calculate the disparitry map
    # grabCut()       # To execute the grabCut alghorithm
    watershed()  # To execute the watershed algorithm
