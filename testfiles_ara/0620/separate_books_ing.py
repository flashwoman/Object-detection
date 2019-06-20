import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def display(winname, img):
    cv.moveWindow(winname, 500, 0)
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.imshow(winname, img)
    cv.waitKey(0)


def watershed(img):
    # Median Blur
    #sep_blur = cv.medianBlur(img, 5)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display('gray', gray)

    # ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY )
    # display('sep_thr', thresh)


    # NOISE REMOVAL (OPTIONAL)
    v_kernel = np.ones((3, 1), np.uint8)
    h_kernel = np.ones((1, 3), np.uint8)


    #위 아래로 연결
    opening = cv.morphologyEx(gray, cv.MORPH_OPEN, v_kernel, iterations=2)
    # ret, thresh = cv.threshold(opening, 50, 255, cv.THRESH_BINARY )
    display('thresh', thresh)



    # 양옆으로 연결해주기
    # opening = cv.morphologyEx(thresh, cv.MORPH_CLOSE, h_kernel, iterations=2)

    # sure_bg = cv.dilate(opening, kernel, iterations=3)
    display('opening', opening)


    # contours
    images,contours, hierarchy = cv.findContours(opening, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv.drawContours(img, contours, i, (255, 0, 0), 1)

    display('sep_coins', img)



def main():
    img = cv.imread('C:/dev/Object-detection/testfiles_ara\output/real_bookshelf_02_fin.jpg')
    #img = cv.imread('C:/dev/Object-detection/testfiles_ara/0618/img/img_book_only.png')
    img_o = img.copy()
    display('img', img)
    # print(img.shape)
    watershed(img)


   # contour(img, img_o)



if __name__ == "__main__":
    main()

