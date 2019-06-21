import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def display(winname, img):
    cv.moveWindow(winname, 500, 0)
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.imshow(winname, img)
    cv.waitKey(0)


def preprocessing(img):

    # vertical kernel
    v_kernel = np.ones((13, 3), np.uint8)
    # morphologyEx
    v_opening = cv.morphologyEx(img, cv.MORPH_OPEN, v_kernel, iterations=2)

    # sure_bg = cv.dilate(opening, kernel, iterations=3)
    display('v_opening', v_opening)

    gray = cv.cvtColor(v_opening, cv.COLOR_BGR2GRAY)
    display('gray', gray)

    ret, thresh = cv.threshold(gray, 10, 255, cv.THRESH_TOZERO)
    display('thresh', thresh)



    # horizontal kernel
    h_kernel = np.ones((3, 13), np.uint8)
    # morphologyEx
    h_opening = cv.morphologyEx(img, cv.MORPH_OPEN, h_kernel, iterations=2)
    display('h_opening', h_opening)


    gray = cv.cvtColor(h_opening, cv.COLOR_BGR2GRAY)
    display('gray', gray)

    ret, thresh = cv.threshold(gray, 10, 255, cv.THRESH_TOZERO)
    display('thresh', thresh)

    bitwise_or = cv.bitwise_or(v_opening, h_opening )
    display('bitwise_or', bitwise_or)


    detected_books = cv.bitwise_and(img, img, mask=bitwise_or)
    display('bitwise_or', detected_books)

    return detected_books

def contours(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display('gray', gray)

    images,contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        # 외곽선 추출
        # if hierarchy[0][i][3] == -1:
        #     cv.drawContours(img, contours, i, (255, 0, 0), 1)
        cv.drawContours(img, contours, i, (255, 0, 0), 1)
    display('contours', img)

    return img



def main():
    img = cv.imread('C:/dev/Object-detection/testfiles_ara/0618/img/img_book_only.png')
    img_o = img.copy()
    display('img', img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display('gray', gray)

    # 1. 배경색과 다른 책들 처리
    # ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY )
    # display('sep_thr', thresh)
    # img = preprocessing(img)
    #

    # # 2. 배경색과 비슷한 책들
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV )
    display('sep_thr', thresh)
    img = preprocessing(img)

    img = contours(img)



if __name__ == "__main__":
    main()

