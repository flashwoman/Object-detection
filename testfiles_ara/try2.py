import numpy as np
import cv2 as cv
from PIL import Image

img = cv.imread('C:/fleshwoman/Object-detection/image/test.jpg', cv.IMREAD_GRAYSCALE)
img_origin = cv.imread('C:/fleshwoman/Object-detection/image/test.jpg')
height, width= img.shape[:2]

#전역 thresholding 적용
ret, thr1 = cv.threshold(img, 127,255,cv.THRESH_BINARY)

#Otsu 바이너리제이션
ret, thr2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

#가우시안 블러 적용 후 Otsu
#blur = cv.GaussianBlur(img, (5,5), 0)
blur = cv.bilateralFilter(img, 9, 50, 50)
ret, thr3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

#Adaptive threshold
thr4 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
thr5 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

# Adaptive threshold + blur
thr6 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, -2)
thr7 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

titles = ['origin', 'His', 'G-thr',
          'thr2', 'His', 'G-thr + Otsu',
          'thr3', 'His', 'G-thr + Blur + Otsu',
          'thr4', 'His', 'A-thr + Mean',
          'thr5', 'His', 'G-thr + Gaus'
          'thr6', 'His', 'G-thr + Mean + blur',
          'thr7', 'His', 'G-thr + Gaus + blur',
          ]
images = [img, 0, thr1,
          img, 0, thr2,
          blur, 0, thr3,
          img, 0, thr4,
          img, 0, thr5,
          img, 0, thr6,
          img, 0, thr7,
          ]

# images_row1 = np.hstack([img, thr1, thr2, thr3])
# images_row2 = np.hstack([thr4, thr5, thr6, thr7])
# res = np.vstack([images_row1, images_row2])




cv.namedWindow('img', cv.WINDOW_NORMAL )
# cv.imshow('img', res)
# cv.waitKey(0)

cv.imshow('img', thr6)
cv.waitKey(0)


# # closing or open
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
# # opened = cv2.morphologyEx(thr7, cv2.MORPH_OPEN, kernel)
# closed = cv.morphologyEx(thr6, cv.MORPH_CLOSE, kernel)
# #closed = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel)
# closed = cv.Canny(closed, 10, 80)
#
# #oc = np.vstack([thr7, closed])
oc = thr6
cv.namedWindow('oc', cv.WINDOW_NORMAL)
cv.imshow("oc", oc)
cv.waitKey(0)



#################### countours ####################
# a = thr6.copy()
# contours, _ = cv.findContours(a, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#
#
# min_length = height/2
# box1 = []
# for cnt in contours :
#     area = cv.contourArea(cnt)
#     x, y, w, h = cv.boundingRect(cnt)
#     if h > min_length:
#         cv.rectangle(img_origin, (x,y), (x+w, y+h), (0,255,0),1 )
#         box1.append(cv.boundingRect(cnt))
#
# fin = img_origin

# # #################### hougphlineP ####################
# minLineLength = 80     # minLineLength : 이 값이하로 떨어진 선 길이는 직선으로 간주하지 않는다.
# maxLineGap = 10        # maxLineGap : 직선이 이 값 이상으로 떨어져 있으면 다른 직선으로 간주한다.
# rho = 1
# theta = np.pi/180
# threshold = 300
#
# min_length = height/2
# print(min_length)
#
# res = img_origin.copy()
# lines = cv.HoughLinesP(oc, 1, theta, threshold, minLineLength, maxLineGap)
# for i in range(len(lines)):
#     for x, y, w, h in lines[i]:
#         print(x,y,w,h)
#         if h > min_length:
#             cv.line(res, (x, y), (w, h), (0, 0, 255), 1)
#
# fin = np.hstack([img_origin, res])

#################### hougphline ####################

# rho = 1
# theta = np.pi/180
# # 이미지에 따라 다름..
# threshold = 100
# lines = cv.HoughLines(closed, 1, theta, threshold)
#
# res = img_origin.copy()
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
#      cv.line(res, (x1, y1), (x2, y2), (255, 0, 0), 2)
#      cropped = res[100:200, 500:640]




#################### Output ####################
cv.namedWindow('fin', cv.WINDOW_NORMAL)
cv.imshow("fin", fin )
cv.waitKey(0)



cv.destroyAllWindows()

