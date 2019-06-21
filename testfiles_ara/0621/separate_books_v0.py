import numpy as np
import cv2 as cv
import imutils


def display(winname, img):
    cv.moveWindow(winname, 500, 0)
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.imshow(winname, img)
    cv.waitKey(0)


def preprocessing(img):

    #1. vertical kernel
    v_kernel = np.ones((11, 2), np.uint8)
    # morphologyEx
    v_opening = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, v_kernel, iterations=2)

    # sure_bg = cv.dilate(opening, kernel, iterations=3)
    #display('v_opening', v_opening)

    v_gray = cv.cvtColor(v_opening, cv.COLOR_BGR2GRAY)
    #display('v_gray', v_gray)

    # ret, v_thresh = cv.threshold(v_gray, 10, 255, cv.THRESH_TOZERO)
    # display('v_thresh', v_thresh)

    kernel = np.ones((2, 1), np.uint8)
    erode  = cv.morphologyEx(v_gray, cv.MORPH_ERODE, kernel, iterations=2)
    display('erode', erode)

    return erode


    # 2. horizontal kernel
    # h_kernel = np.ones((2, 11), np.uint8)
    # # morphologyEx
    # h_opening = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, h_kernel, iterations=2)
    # display('h_opening', h_opening)
    #
    # h_gray = cv.cvtColor(h_opening, cv.COLOR_BGR2GRAY)
    # display('h_gray', h_gray)

    # ret, h_thresh = cv.threshold(h_gray, 10, 255, cv.THRESH_TOZERO)
    # display('h_thresh', h_thresh)

    # bitwise_or = cv.bitwise_or(v_opening, h_opening )
    # display('bitwise_or', bitwise_or)
    #
    # gray_bitwise_or = cv.bitwise_or(v_gray, h_gray )
    # display('h_bitwise_or', gray_bitwise_or)

    # detected_books = cv.bitwise_and(img, img, mask=bitwise_or)
    # display('bitwise_or', detected_books)
    #
    # return bitwise_or

def contours(img, img_color):

    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # display('gray', gray)
    #
    ## CANNY
    # canny_img = cv.Canny(img, 127, 80)  # 작은게 더 좋으넹 (이진화기준 값을 너무 높게 주지 말 것!)
    # display('canny_img', canny_img)

    ret, img_binary = cv.threshold(img, 127, 255, 0)
    display("img_binary", img_binary)

    imgs,contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    img_color1 = img_color.copy()
    # for cnt in contours:
    #     cv.drawContours(img_color, [cnt], 0, (255, 0, 0), 1)  # blue
    #
    # display("rect", img_color)

    img = cv.imread('C:/dev/Object-detection/EyeCandy/img/img_book_only.png')
    count = 0
    for cnt in contours:
        epsilon = 0.02 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        cv.approxPolyDP((cnt, ))
        cv.drawContours(img_color, [approx], 0, (0, 255, 0 ), 1)

    display("result", img_color)


    # images,contours, hierarchy = cv.findContours(canny_img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    # for i in range(len(contours)):
    #     # 외곽선 추출
    #     #if hierarchy[0][i][3] == -1:
    #     #    cv.drawContours(img_o, contours, i, (255, 0, 0), 1)
    #     cv.drawContours(img_o, contours, i, (255, 0, 0), 1)
    # display('contours', img_o)

    return img



def main():
    img = cv.imread('C:/dev/Object-detection/EyeCandy/img/img_book_only.png')
    img_o = img.copy()
    display('img', img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display('gray', gray)

    # 전처리
    img = preprocessing(img)

    #contours
    img = contours(img,img_o)



if __name__ == "__main__":
    main()

