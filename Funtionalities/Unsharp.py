import cv2 as cv


def unsharpMasking(img):
    """
    Unsharp masking.
    Steps:
    - blur an image
    - difference between blur and orginal image
    - add blur image with weight mask
    """

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)
    mask = gray - blurred

    alpha = 1
    unsharp = gray + alpha * mask

    # Other way to do it.
    '''
    channels = cv2.split(gray)
    channelsMask = cv2.split(mask)

    for i, channel in enumerate(channels):
        channel[:] += channelsMask[i] * alpha

    unsharp = cv2.merge(channels)
    '''

    return gray, unsharp


img = cv.imread("../Imgs/statue.jpg")
img = cv.resize(img, (500, 500))
gray, unsharp = unsharpMasking(img.copy())

cv.imshow("Original", gray)
cv.imshow("Unsharp", unsharp)

cv.waitKey()
cv.destroyAllWindows()

