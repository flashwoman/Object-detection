
import cv2
import numpy as np

img = cv2.imread('C:/fleshwoman/Object-detection/image/test.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges1 = cv2.Canny(gray,50,150,apertureSize = 3)

cv2.namedWindow('edges1', cv2.WINDOW_NORMAL)
cv2.imshow('edges1',edges1)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
closed = cv2.morphologyEx(edges1, cv2.MORPH_CLOSE, kernel)
# closed = cv2.GaussianBlur(closed, (3, 3), 0)

cv2.namedWindow('closed', cv2.WINDOW_NORMAL)
cv2.imshow('closed',closed)
cv2.waitKey(0)

ret, thr = cv2.threshold(closed, 127,255,0)

contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i in range(0,len(contours)):
    cnt = contours[i]
    cv2.drawContours(img,[cnt],0,(255,255,0), 1)



cv2.namedWindow('output.jpg', cv2.WINDOW_NORMAL)
cv2.imshow('output.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#
#
# minLineLength = 200
# maxLineGap = 100
# rho = 1
# theta = np.pi/180 * 90
# threshold = 80
# lines = cv2.HoughLinesP(closed, 1, theta, threshold, minLineLength, maxLineGap)
#
# for i in range(len(lines)):
#     for x1,y1,x2,y2 in lines[i]:
#         cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
#
#
#
# cv2.imshow('output.jpg',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()