import numpy as np
import cv2 as cv
from PIL import Image

def thresholding():
    img_o = cv.imread('C:/fleshwoman/Object-detection/image/real_bookshelf_02.jpg', cv.IMREAD_LOAD_GDAL)
    img = cv.imread('C:/fleshwoman/Object-detection/image/real_bookshelf_02.jpg', cv.IMREAD_GRAYSCALE)
    # cv.IMREAD_LOAD_GDAL
    # real_bookshelf_02_fin_36.jpg

    a = np.hstack([img_o])
    cv.namedWindow('a', cv.WINDOW_NORMAL)
    cv.imshow('a', a)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #전역 thresholding 적용
    ret, thr1 = cv.threshold(img, 127,255,cv.THRESH_BINARY)

    #Otsu 바이너리제이션
    ret, thr2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU )

    #가우시안 블러 적용 후 Otsu
    blur = cv.GaussianBlur(img, (11,11), 0)
    ret, thr3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    #Adaptive threshold
    thr4 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    thr5 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    # Adaptive threshold + blur
    thr6 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 33, 5)
    thr7 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 33, 2)

    ret, thr8 = cv.threshold(img, 10, 255, cv.THRESH_TRUNC )

    res = np.hstack([thr6, thr8])

    cv.namedWindow('res', cv.WINDOW_NORMAL)
    cv.imshow('res', res)
    cv.waitKey(0)
    cv.destroyAllWindows()


    print(thr6.shape)
#
#     titles = ['origin', 'His', 'G-thr',
#               'thr2', 'His', 'G-thr + Otsu',
#               'thr3', 'His', 'G-thr + Blur + Otsu',
#               'thr4', 'His', 'A-thr + Mean',
#               'thr5', 'His', 'G-thr + Gaus'
#               'thr6', 'His', 'G-thr + Mean + blur',
#               'thr7', 'His', 'G-thr + Gaus + blur',
#               ]
#     images = [img, 0, thr1,
#               img, 0, thr2,
#               blur, 0, thr3,
#               img, 0, thr4,
#               img, 0, thr5,
#               img, 0, thr6,
#               img, 0, thr7,
#               ]
#
#
#     images_row1 = np.hstack([img, thr1] )
#     images_row2 = np.hstack([thr2, thr3])
#
#     images_row3 = np.hstack([thr4, thr5])
#     images_row4 = np.hstack([thr6, thr7])
#
#     cv.namedWindow('img0', cv.WINDOW_NORMAL )
#     cv.imshow('img0', img_o)
#
#
#     cv.namedWindow('img2', cv.WINDOW_NORMAL )
#     cv.imshow('img2', images_row2)
#
#     cv.namedWindow('img3', cv.WINDOW_NORMAL )
#     cv.imshow('img3', images_row3)
#
#     cv.namedWindow('img4', cv.WINDOW_NORMAL )
#     cv.imshow('img4', images_row4)
#
#     fin1 = cv.addWeighted( src1=thr3, alpha=0.5, src2=thr6, beta=0.5, gamma=0 )
#     #파일저장
#     path = "real_bookshelf_02_fin_36.jpg"
#     cv.imwrite(path, fin1)
#
#     fin2 = cv.addWeighted( src1=thr3, alpha=0.5, src2=thr4, beta=0.5, gamma=0 )
#     #파일저장
#     path = "real_bookshelf_02_fin_34.jpg"
#     cv.imwrite(path, fin2)
#
#     cv.namedWindow('fin2', cv.WINDOW_NORMAL)
#     cv.imshow('fin2', fin2)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
# #





thresholding()