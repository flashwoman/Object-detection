
import numpy as np
import cv2

img_original = cv2.imread('C:/fleshwoman/Object-detection/image/books.jpg')

img = img_original.copy()


# 1. 이미지 가공하기
## 1.1 GrayScale로 바꾸고 Blur 처리
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur1 = cv2.GaussianBlur(gray, (5, 5), 0)
#edge 보존 but 느리다. 화질 좋을때 text가 잘 잡힘
blur2 = cv2.bilateralFilter(gray, 9, 75, 75)
# src: 입력영상
# d : 필터링에 이용하는 이웃한 픽셀의 지름을 정의, 불가능한 경우 sigmaSpace를 사용함.
# sigmaColor : 컬러공간의 시그마공간 정의, 클수록 이웃한 픽셀과 기준색상의 영향이 커진다.
# sigmaSpace : 시그마 필터를 조절한다, 값이 클수록 긴밀하게 주변 픽셀에 영향을 미친다. d > 0 이면 영향을 받지 않고, 그 이외에는 d 값에 비례한다.
#

blur3 = cv2.medianBlur(gray, 5)

# blur = np.hstack([blur1, blur2, blur3])
# cv2.namedWindow('blur', cv2.WINDOW_NORMAL)
# cv2.imshow("blur", blur)
# cv2.waitKey(0)



# 1.2. Threshold 적용
# Threshoding.py에서 확인결과 가장 적합해 보이는 코드
thr7 = cv2.adaptiveThreshold(blur3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -2)
# cv2.namedWindow('thr', cv2.WINDOW_NORMAL)
# cv2.imshow("thr", thr7)
# cv2.waitKey(0)

# closing or open
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# opened = cv2.morphologyEx(thr7, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(thr7, cv2.MORPH_CLOSE, kernel)

#oc = np.vstack([thr7, closed])
oc = closed
cv2.namedWindow('oc', cv2.WINDOW_NORMAL)
cv2.imshow("oc", oc)
cv2.waitKey(0)

# 2. Edge 검출
# edges = cv2.Canny(blur, 5, 250)
# cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
# cv2.imshow("edges", edges)
# cv2.waitKey(0)
#
#minLineLength : 이 값이하로 떨어진 선 길이는 직선으로 간주하지 않는다.
minLineLength = 100
#maxLineGap : 직선이 이 값 이상으로 떨어져 있으면 다른 직선으로 간주한다.
maxLineGap = 10
rho = 1
theta = np.pi/180
threshold = 200


lines = cv2.HoughLinesP(oc, 1, theta, threshold, minLineLength, maxLineGap)
lines1 = cv2.HoughLinesP(oc, 1, theta, threshold, 100, 5)
lines2 = cv2.HoughLinesP(oc, 1, theta, threshold, 100, maxLineGap)
lines3 = cv2.HoughLinesP(oc, 1, theta, 150, 100, 5)

img1 = img.copy()
img2 = img.copy()
img3 = img.copy()


for i in range(len(lines)):
    for x1, y1, x2, y2 in lines[i]:
        cv2.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 3)

for i in range(len(lines1)):
    for x1, y1, x2, y2 in lines1[i]:
        cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 3)

for i in range(len(lines2)):
    for x1, y1, x2, y2 in lines2[i]:
        cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 3)

for i in range(len(lines3)):
    for x1, y1, x2, y2 in lines3[i]:
        cv2.line(img3, (x1, y1), (x2, y2), (0, 0, 255), 3)



row1 = np.hstack([img, img1])
row2 = np.hstack([img2, img3])
fin = np.vstack([row1, row2])

cv2.namedWindow('fin', cv2.WINDOW_NORMAL)
cv2.imshow('fin', fin)
cv2.waitKey(0)
cv2.destroyAllWindows()




# minLineLength = 500
# maxLineGap = 10
# rho = 1
# # np.pi/180 : 2도 단위이므로 각 픽셀별로 180개의 r,theta가 나온다.
# theta = np.pi/45
# # 이미지에 따라 다름..
# threshold = 100
# lines = cv2.HoughLines(closed, 1, theta, threshold)
#
# for line in lines:
#     for rho,theta in line:
#      a = np.cos(theta)
#      b = np.sin(theta)
#      x0 = a*rho
#      y0 = b*rho
#      x1 = int(x0 + 1000*(-b))
#      y1 = int(y0 + 1000*(a))
#      x2 = int(x0 - 1000*(-b))
#      y2 = int(y0 - 1000*(a))
#      cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#      cropped = img[100:200, 500:640]
#
#
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
