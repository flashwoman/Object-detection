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


img_origin = cv2.imread('C:/fleshwoman/Object-detection/image/test.jpg')
img = img_origin.copy()
height, width= img.shape[:2]    # [height, width, channel] = img.shape
print(height, width)

ratio = img.shape[1] / 900.0
img = imutils.resize(img, width=900)


#노이즈제거 및 edge
gray_img = grayscale(img)
blur_img = bilateralFilter(gray_img, 5, 100)
canny_img = canny(blur_img, 10, 110) # 작은게 더 좋으넹 (이진화기준 값을 너무 높게 주지 말 것!)

####test#####
# canny_img1 = canny(blur_img, 50, 210)
# canny_img2 = canny(blur_img, 10, 400)
# canny_img3 = canny(blur_img, 50, 400)
# row1 = np.hstack([canny_img, canny_img1])
# row2 = np.hstack([canny_img2, canny_img3])
# canny = np.vstack([row1,row2])
#

cv2.imshow("canny_img", canny_img )
cv2.waitKey(0)


#canny = gaussian_blur(canny_img, 9)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, kernel)

cv2.imshow("closed", closed )
cv2.waitKey(0)