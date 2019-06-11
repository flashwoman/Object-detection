import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# using sobel()
path = "C:/Users/DELL/PycharmProjects/Object-detection/image/bookshelf_04.jpg"
img = cv.imread(path, cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

tmp = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=5)
sobel64f = np.absolute(tmp)
sobel_united = np.uint8(sobel64f)

cannied_max = cv.Canny(img_gray, 250, 500)
cannied_mean = cv.Canny(img_gray, 50, 500)
# cannied_min = cv.Canny(img_gray, 150, 200)

# MorphologyEx (노이즈 제거하기)
# 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
img_mask1 = cv.inRange(img_hsv, lower_color1, upper_color1)
img_mask2 = cv.inRange(img_hsv, lower_color2, upper_color2)
img_mask3 = cv.inRange(img_hsv, lower_color3, upper_color3)
img_mask = img_mask1 | img_mask2 | img_mask3
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))

img_mask = cv.morphologyEx(img_mask, cv.MORPH_OPEN, kernel)
img_mask = cv.morphologyEx(img_mask, cv.MORPH_CLOSE, kernel)

# plt.imshow(sobel_united, cmap='gray')
plt.title('sobel_using 64F')
plt.xticks([])
plt.yticks([])
plt.show()

cv.imshow('canny_max', cannied_max)
cv.imshow('canny_mean', cannied_mean)
# cv.imshow('canny_min', cannied_min)
cv.waitKey(0)
cv.destroyAllWindows()