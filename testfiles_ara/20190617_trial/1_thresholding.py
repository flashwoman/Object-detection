import numpy as np
import cv2 as cv
from PIL import Image

def thresholding():
    img = cv.imread('C:/fleshwoman/Object-detection/image/real_bookshelf_02.jpg', cv.IMREAD_GRAYSCALE)

    #전역 thresholding 적용
    ret, thr1 = cv.threshold(img, 127,255,cv.THRESH_BINARY)

    #Otsu 바이너리제이션
    ret, thr2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU )

    #가우시안 블러 적용 후 Otsu
    blur = cv.GaussianBlur(img, (5,5), 0)
    ret, thr3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    #Adaptive threshold
    thr4 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    thr5 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    # Adaptive threshold + blur
    thr6 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 5)
    thr7 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    titles = ['origin', 'His', 'G-thr',
              'thr2', 'His', 'G-thr + Otsu',
              'thr3', 'His', 'G-thr + Blur + Otsu',
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

    # for i in range(6):
    #     plt.subplot(7, 3, i*3+1), plt.imshow(images[i*3], 'gray')
    #     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    #
    #     plt.subplot(7, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
    #     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    #
    #     plt.subplot(7, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
    #     #plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    #
    #     print(i)

    images_row1 = np.hstack([img, thr1, thr2, thr3])
    images_row2 = np.hstack([thr4, thr5, thr6, thr7])
    res = np.vstack([images_row1, images_row2])

    cv.namedWindow('img', cv.WINDOW_NORMAL )
    cv.imshow('img', res)


    fin2 = cv.addWeighted( src1=thr2, alpha=0.2, src2=thr6, beta=0.8, gamma=0 )

    cv.namedWindow('fin2', cv.WINDOW_NORMAL)
    cv.imshow('fin2', fin2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    #파일저장
    path = "C:/fleshwoman/Object-detection-dev/output/real_bookshelf_02_fin.jpg"
    cv.imwrite(path, fin2)


thresholding()