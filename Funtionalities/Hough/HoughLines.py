import cv2
import numpy as np

img = cv2.imread('../../Imgs/Hough/duomo.jpg')
#img = cv2.imread('../../Imgs/metro.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 120)
minLineLength = 20
maxLineGap = 5

lines = cv2.HoughLinesP(edges, 1, np.pi/180.0, 20, minLineLength, maxLineGap)
for x1, y1, x2, y2 in lines[0]:
    cv2.line(img, (x1, y1), (x2, y2), (0,255,0),2)

cv2.imshow("Edges", edges)
cv2.imshow("Lines", img)

cv2.waitKey()
cv2.destroyAllWindows()
