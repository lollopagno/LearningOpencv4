import cv2
import numpy as np


def edge_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rect = cv2.minAreaRect(c)  # Minimal rectangar area
        box = cv2.boxPoints(rect)  # Find 4 vertices of the rectangle
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

        (x, y), radius = cv2.minEnclosingCircle(c)
        center, radius = (int(x), int(y)), int(radius)
        img = cv2.circle(img, center, radius, (0, 255, 0), 2)

    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

    return img


def convex_countour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img)
    for c in contours:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        hull = cv2.convexHull(c)

        cv2.drawContours(mask, [c], -1, (255, 0, 0), 1)
        cv2.drawContours(mask, [approx], -1, (0, 255, 0), 1)
        cv2.drawContours(mask, [hull], -1, (0, 0, 255), 1)

    return mask


if __name__ == "__main__":
    img = cv2.pyrDown(cv2.imread("../Imgs/hammer.jpg", cv2.IMREAD_UNCHANGED))

    edge = edge_detector(img.copy())
    hull = convex_countour(img.copy())

    cv2.imshow("Orginal", img)
    cv2.imshow("Contours", edge)
    cv2.imshow("Hull", hull)

    cv2.waitKey()
    cv2.destroyAllWindows()
