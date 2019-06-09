import numpy as np
import cv2
import imutils
from PIL import Image

# GrayScale
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#Canny
def canny(img, low_thr, high_thr):
    return cv2.Canny(img, low_thr, high_thr)

def gaussian_blur(img,  kernel_size):
    return cv2.GaussianBlur(img,  (kernel_size,kernel_size), 0)

def bilateralFilter(img,  d, sigma):
    # src: 입력영상
    # d : 필터링에 이용하는 이웃한 픽셀의 지름을 정의, 불가능한 경우 sigmaSpace를 사용함.
    # sigmaColor : 컬러공간의 시그마공간 정의, 클수록 이웃한 픽셀과 기준색상의 영향이 커진다.
    # sigmaSpace : 시그마 필터를 조절한다, 값이 클수록 긴밀하게 주변 픽셀에 영향을 미친다. d > 0 이면 영향을 받지 않고, 그 이외에는 d 값에 비례한다.
    return cv2.bilateralFilter(img, d, sigma, sigma)

def morphologyEx(img, size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, kernel)

def findContours(img):
    return cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


img_origin = cv2.imread('C:/fleshwoman/Object-detection/image/2sCuq.jpg')
img = img_origin.copy()

# resize
ratio = img.shape[1] / 900.0
img = imutils.resize(img, width=900)
height, width= img.shape[:2]    # [height, width, channel] = img.shape

#노이즈제거 및 edge
gray_img = grayscale(img)
#blur_img1 = bilateralFilter(img, 5, 100)
blur_img = gaussian_blur(img, 5)

#blur_img = np.hstack([blur_img,blur_img1])
cv2.imshow("blur_img", blur_img)
cv2.waitKey(0)

canny_img = canny(blur_img, 10, 80) # 작은게 더 좋으넹 (이진화기준 값을 너무 높게 주지 말 것!)

cv2.imshow("canny_img", canny_img )
cv2.waitKey(0)


# canny = gaussian_blur(canny_img, 9)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5 , 5))
closed = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, kernel)

cv2.imshow("closed", closed )
cv2.waitKey(0)


#################################################################################################################################3

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cc = closed
cnts = cv2.findContours(canny_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

print(len(cnts))
# loop over the contours

########################################### contour 검출 #########################
#
# zero = np.zeros(img.shape)
# img_Area = height * width
#
# for c in cnts:
#
#     peri = cv2.arcLength(c, True)   # 둘레
#     area = cv2.contourArea(c)       # 넓이
#
#     # https://m.blog.naver.com/samsjang/220516822775 : 두번째 인자의 값을 기준으로 contour를 따라 대략적인 도형을 만들어줌
#     #approx = cv2.approxPolyDP(c, 0.01 * peri, True) # 0.01이 적당 (test.jpg 기준)
#
#     cv2.drawContours(zero,cnts,-1, (255,255,255), 1)
#
#     if ( len(approx) >=8 )  :
#         print(peri, area)
#         screenCnt = approx
#         cv2.drawContours(zero, [screenCnt], -1, (255, 255, 255), 1)
#
# cv2.imshow("Outline", zero)
# cv2.waitKey(0)
#
# print('width :' , width)


########################################### text 제거 #########################

zero = np.zeros(img.shape)
img_Area = height * width

for c in cnts:

    peri = cv2.arcLength(c, True)   # 둘레
    area = cv2.contourArea(c)       # 넓이

    c = c.astype("float")
    c = c.astype("int")

    x, y, w, h = cv2.boundingRect(c)

    if   h < height / 2 :
        continue

    cv2.rectangle(img, (x, y), (x + w, y + h), (3, 255, 4), 2)

cv2.imshow("image", img)
cv2.waitKey(0)


########################################### line 검출 #########################
rho = 1
# np.pi/180 : 2도 단위이므로 각 픽셀별로 180개의 r,theta가 나온다.
theta = np.pi/180

# 이미지에 따라 다름..
# test.jpg, canny_img는 165가 적당.
threshold = 165


lines = cv2.HoughLines(canny_img, 1, theta, threshold)

for line in lines:

    for rho,theta in line:
     a = np.cos(theta)
     b = np.sin(theta)

     x0 = a*rho
     y0 = b*rho

     x1 = int(x0 + 1000*(-b))
     y1 = int(y0 + 1000*(a))
     x2 = int(x0 - 1000*(-b))
     y2 = int(y0 - 1000*(a))

     if ( abs(x2 - x1) < width/ 9) :
         print(x1, y1,x2, y2)
         cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
     #cropped = img[100:200, 500:640]


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()






# #
# #minLineLength : 이 값이하로 떨어진 선 길이는 직선으로 간주하지 않는다.
# minLineLength = 500
# #maxLineGap : 직선이 이 값 이상으로 떨어져 있으면 다른 직선으로 간주한다.
# maxLineGap = 5
#
# rho = 1
# theta = np.pi/180
# threshold = 150
#
#
# lines = cv2.HoughLinesP(canny_img, 1, theta, threshold, minLineLength, maxLineGap)
# lines1 = cv2.HoughLinesP(canny_img, 1, theta, threshold, 100, 5)
# lines2 = cv2.HoughLinesP(canny_img, 1, theta, threshold, 100, maxLineGap)
# lines3 = cv2.HoughLinesP(canny_img, 1, theta, 100, 100, 5)
#
# img1 = img.copy()
# img2 = img.copy()
# img3 = img.copy()
#
#
# for i in range(len(lines)):
#     for x1, y1, x2, y2 in lines[i]:
#         cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
#
# for i in range(len(lines1)):
#     for x1, y1, x2, y2 in lines1[i]:
#         cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 3)
#
# for i in range(len(lines2)):
#     for x1, y1, x2, y2 in lines2[i]:
#         cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 3)
#
# for i in range(len(lines3)):
#     for x1, y1, x2, y2 in lines3[i]:
#         cv2.line(img3, (x1, y1), (x2, y2), (0, 0, 255), 3)
#
#
#
# row1 = np.hstack([img, img1])
# row2 = np.hstack([img2, img3])
# fin = np.vstack([row1, row2])
#
# cv2.namedWindow('fin', cv2.WINDOW_NORMAL)
# cv2.imshow('fin', fin)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
















