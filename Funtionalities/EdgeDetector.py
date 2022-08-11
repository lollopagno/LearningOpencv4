import cv2 as cv
import numpy as np


def edge_detector(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rect = cv.minAreaRect(c)  # Minimal rectangar area
        box = cv.boxPoints(rect)  # Find 4 vertices of the rectangle
        box = np.int0(box)
        cv.drawContours(img, [box], 0, (0, 0, 255), 3)

        (x, y), radius = cv.minEnclosingCircle(c)
        center, radius = (int(x), int(y)), int(radius)
        img = cv.circle(img, center, radius, (0, 255, 0), 2)

    cv.drawContours(img, contours, -1, (255, 0, 0), 1)

    return img


def convex_countour(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img)
    for c in contours:
        epsilon = 0.01 * cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, epsilon, True)
        hull = cv.convexHull(c)

        cv.drawContours(mask, [c], -1, (255, 0, 0), 1)
        cv.drawContours(mask, [approx], -1, (0, 255, 0), 1)
        cv.drawContours(mask, [hull], -1, (0, 0, 255), 1)

    return mask


if __name__ == "__main__":
    img = cv.pyrDown(cv.imread("../Imgs/hammer.jpg", cv.IMREAD_UNCHANGED))

    edge = edge_detector(img.copy())
    hull = convex_countour(img.copy())

    cv.imshow("Orginal", img)
    cv.imshow("Contours", edge)
    cv.imshow("Hull", hull)

    cv.waitKey()
    cv.destroyAllWindows()
