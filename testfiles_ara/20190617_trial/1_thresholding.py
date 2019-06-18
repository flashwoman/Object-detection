import numpy as np
import cv2 as cv
from PIL import Image

def thresholding():
    img_o = cv.imread('C:/fleshwoman/Object-detection/image/real_bookshelf_02.jpg')
    img = cv.imread('C:/fleshwoman/Object-detection/image/real_bookshelf_02.jpg', cv.IMREAD_GRAYSCALE)

    gray = cv.imread('real_bookshelf_02_fin_36.jpg', cv.IMREAD_GRAYSCALE)


    blur = cv.GaussianBlur(gray, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(blur, cv.MORPH_OPEN, kernel)

    cv.namedWindow('opening', cv.WINDOW_NORMAL)
    cv.imshow('opening', opening)
    cv.waitKey(0)

    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
    # return cv.morphologyEx(canny_img, cv.MORPH_CLOSE, kernel)

    #전역 thresholding 적용
    ret, thr1 = cv.threshold(img, 127,255,cv.THRESH_BINARY)

    #Otsu 바이너리제이션
    ret, thr2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU )

    #가우시안 블러 적용 후 Otsu
    ret, thr3 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    #Adaptive threshold
    thr4 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    thr5 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    # Adaptive threshold + blur
    thr6 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 5)
    thr7 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    titles = ['origin', 'His', 'G-thr',
              'thr2', 'His', 'G-thr + Otsu',
              'thr3', 'His', 'G-th5r + Blur + Otsu',
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


    images_row1 = np.hstack([img, thr1] )
    images_row2 = np.hstack([thr2, thr3])

    images_row3 = np.hstack([thr4, thr5])
    images_row4 = np.hstack([thr6, thr7])

    cv.namedWindow('img0', cv.WINDOW_NORMAL )
    cv.imshow('img0', img_o)


    cv.namedWindow('img2', cv.WINDOW_NORMAL )
    cv.imshow('img2', images_row2)

    cv.namedWindow('img3', cv.WINDOW_NORMAL )
    cv.imshow('img3', images_row3)

    cv.namedWindow('img4', cv.WINDOW_NORMAL )
    cv.imshow('img4', images_row4)


#################################### addWeignt
    fin1 = cv.addWeighted( src1=thr3, alpha=0.4, src2=thr6, beta=0.6, gamma=0 )
    #파일저장
    path = "real_bookshelf_02_fin_362.jpg"
    cv.imwrite(path, fin1)

    fin2 = cv.addWeighted( src1=thr3, alpha=0.4, src2=thr4, beta=0.6, gamma=0 )
    #파일저장
    path = "real_bookshelf_02_fin_342.jpg"
    cv.imwrite(path, fin2)

    cv.namedWindow('fin2', cv.WINDOW_NORMAL)
    cv.imshow('fin2', fin2)
    cv.waitKey(0)

    fin3 = cv.addWeighted( src1=thr2, alpha=0.4, src2=thr6, beta=0.6, gamma=0 )
    #파일저장
    path = "real_bookshelf_02_fin_262.jpg"
    cv.imwrite(path, fin3)

###################################### bitWise
#1 흰배경 and
    w_and = cv.bitwise_and( thr2, thr6 )

#2 흰배경 or
    w_or = cv.bitwise_or(thr2, thr6)
#3 흰배경 xor
    w_xor = cv.bitwise_xor(thr2, thr6)

#1 검은배경 and
    b_and = cv.bitwise_and(thr3, thr6)
#2 검은배경 or
    b_or = cv.bitwise_or(thr3, thr6)
#3 검은배경 xor
    b_xor = cv.bitwise_xor(thr3, thr6)

    images_w = np.hstack([w_and, w_or, w_xor ])
    images_b = np.hstack([b_and, b_or, b_xor])

    image_not = cv.bitwise_not(b_and)


    cv.namedWindow('images_w', cv.WINDOW_NORMAL)
    cv.imshow('images_w', images_w)
    cv.waitKey(0)

    cv.namedWindow('images_b', cv.WINDOW_NORMAL)
    cv.imshow('images_b', images_b)
    cv.waitKey(0)

    cv.namedWindow('image_not', cv.WINDOW_NORMAL)
    cv.imshow('image_not', image_not)
    cv.waitKey(0)

    path = "images_w.jpg"
    cv.imwrite(path, images_w)

    path = "images_b.jpg"
    cv.imwrite(path, images_b)

    # a = cv.bitwise_and(fin2, b_xor)
    #
    # gray = np.hstack([fin2, a])
    #
    # cv.namedWindow('gray', cv.WINDOW_NORMAL)
    # cv.imshow('gray', gray)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


    path = "w_and.jpg"
    cv.imwrite(path, w_and)

    path = "w_or.jpg"
    cv.imwrite(path, w_or)

    path = "w_xor.jpg"
    cv.imwrite(path, w_xor)

    path = "b_and.jpg"
    cv.imwrite(path, b_and)

    path = "b_or.jpg"
    cv.imwrite(path, b_or)

    path = "b_xor.jpg"
    cv.imwrite(path, b_xor)

    path = "b_and_not.jpg"
    cv.imwrite(path, image_not)



thresholding()