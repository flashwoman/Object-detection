import numpy as np
import cv2
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


img_origin = cv2.imread('C:/fleshwoman/Object-detection/testfiles_ara/20190617_trial/real_bookshelf_02_fin_36.jpg')
img = img_origin.copy()
#height, width, channel= img.shape
height, width= img.shape[:2]
print(height, width)

############# logics #################

min_length = height/2

# 1. 노이즈제거
gray_img = grayscale(img)
blur_img = gaussian_blur(gray_img, 5)
#blur_img = bilateralFilter(gray_img, 5, 100)

# 2. text제거
## 2-1. edge 검출
canny_img = canny(blur_img, 70, 210)

## 2-2 이제 텍스트를 뭉개버리자
close_img = morphologyEx(canny_img, 5)


fin = np.hstack([blur_img, close_img])
cv2.namedWindow('fin', cv2.WINDOW_NORMAL)
cv2.imshow("fin", fin )
cv2.waitKey(0)

# 2-3 contour로 영역잡고
contours, _ = findContours(close_img)
box1 = []
for cnt in contours :
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)

    if y+h < min_length:
        cv2.drawContours(img, cnt, -1, (0, 255, 0), 3)
        #cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),1 )
        #box1.append(cv2.boundingRect(cnt))



# 2-4 해당 영역 제거 BaaaaMmmm



# fin = np.hstack([img,close_img])
cv2.namedWindow('fin2', cv2.WINDOW_NORMAL)
cv2.imshow("fin2", img )
cv2.waitKey(0)
cv2.destroyAllWindows()






