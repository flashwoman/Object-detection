# OpenCV provides 2-different HoughLines options (HoughLinesP, HoughLines)

# HoughLinesP Code

import cv2 as cv
import numpy as np

img = cv.imread('hallway.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150)
lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)

for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv.imshow("result", img)
cv.waitKey(0)



# HoughLines Code

img = cv.imread('hallway.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,50,150)
lines = cv.HoughLines(edges,1,np.pi/180,200)

for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv.imshow("result", img)
cv.waitKey(0)



# Hough Circle - Edge의 Gradient Contrast를 이용하여 검출하는 방법으로 접근하기(빠르고 가볍다)

circles = cv.HoughCircles( image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]])
# Variable Description
# - image : 입력이미지, Grayscale only
# - method : HOUGH_GRADIENT 이용하기
# - dp : 입력 이미지를 얼마나 축소할지(반비례 배율)
# - minDist : 검출할 원 사이의 최소 거리
# - circles : 발견한 원 벡터값 (x, y, radius, (votes))
# - param1,2,3 : 지정한 원 검출 방법을 위한 파라미터
# - minRadius/maxRadius : 검출할 원의 최소, 최대 반지름

img = cv.imread('input.jpg',0)
img = cv.medianBlur(img,5)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=35,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv.imshow('detected circles',cimg)
cv.waitKey(0)
cv.destroyAllWindows()
