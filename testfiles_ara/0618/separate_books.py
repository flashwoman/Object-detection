import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# GrayScale
def grayscale(img):
    if len(img.shape) != 2:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img
    return gray

#Canny
def canny(img, low_thr, high_thr):
    return cv.Canny(img, low_thr, high_thr)



def bilateralFilter(img,  d, sigma):
    # src: 입력영상
    # d : 필터링에 이용하는 이웃한 픽셀의 지름을 정의, 불가능한 경우 sigmaSpace를 사용함.
    # sigmaColor : 컬러공간의 시그마공간 정의, 클수록 이웃한 픽셀과 기준색상의 영향이 커진다.
    # sigmaSpace : 시그마 필터를 조절한다, 값이 클수록 긴밀하게 주변 픽셀에 영향을 미친다. d > 0 이면 영향을 받지 않고, 그 이외에는 d 값에 비례한다.
    return cv.bilateralFilter(img, d, sigma, sigma)

def morphologyEx(img, size):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
    return cv.morphologyEx(canny_img, cv.MORPH_CLOSE, kernel)

def findContours(img):
    return cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

def display(winname, img):
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.imshow(winname, img)
    cv.waitKey(0)

def erode(img, size):
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (size, 1))
    # Apply morphology operations
    erode = cv.erode(img, verticalStructure)
    #dilate = cv.dilate(thr4, verticalStructure)
    return erode



def main():
    img = cv.imread('C:/dev/Object-detection/0618/img/img_book_only.png')

    #r기본 size정보 저장
    display('img', img)
    height, width = img.shape[:2]  # [height, width, channel] = img.shape

    # 1. gray scale
    gray = grayscale(img)
    display('gray', gray)

    # 2. thresholding
    blur = cv.GaussianBlur(gray, (1, 1), 0)
    ret, thr1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    # ret, thr1 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY )
    # thr = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 11)
    display('thr1', thr1)
    # display('thr', thr)

    erd = erode( gray, 11)
    display('erd', erd)







    cv.erode(  )








if __name__ == "__main__":
    main()
