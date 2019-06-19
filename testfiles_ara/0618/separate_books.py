import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def display(winname, img):
    cv.moveWindow(winname, 300, 0)
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.imshow(winname, img)
    cv.waitKey(0)


# def morphologyEx(img, row_size, col_size ):
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (row_size, col_size))
#     return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

# def separate(img):
#
#     # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     #
#     # ret, thr1 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU )
#     # thr2 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 21)
#     #
#     # fin = cv.addWeighted( src1=thr1, alpha=0.4, src2=thr2, beta=0.6, gamma=0 )
#     # display('fin', fin)
#     #
#     # blur2 = cv.GaussianBlur(fin, (7, 7), 0)
#     # display('blur2', blur2)
#     #
#     # closed = morphologyEx(blur2, 5,5)
#     # display('closed', closed)

def watershed(img):
    # Median Blur
    #sep_blur = cv.medianBlur(img, 5)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display('gray', gray)

    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY )
    display('sep_thr', thresh)


    # NOISE REMOVAL (OPTIONAL)
    v_kernel = np.ones((13, 2), np.uint8)
    h_kernel = np.ones((1, 3), np.uint8)

    #위 아래로 연결된거 끊기
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, v_kernel, iterations=2)

    # 양옆으로 연결해주기
    #opening = cv.morphologyEx(thresh, cv.MORPH_CLOSE, h_kernel, iterations=2)
    # sure_bg = cv.dilate(opening, kernel, iterations=3)
    display('opening', opening)

    # 배경에서 객제까지의 거리
    # dist_transform = cv.distanceTransform(thresh, cv.DIST_L2, 5)
    # result = cv.normalize(dist_transform, None, 255, 0, cv.NORM_MINMAX, cv.CV_8UC1)
    # display('dist_transform', result)
    #
    # # dist_transform.max() 이용해서 원하는 영역만 ( 붙어있는 애들 띄어내서 더미로 분리)
    # ret, sure_fg = cv.threshold(dist_transform, 0.15
    #                             * dist_transform.max(), 255, 0)
    # display('sure_fg', sure_fg)
    #
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv.subtract(sure_bg, sure_fg)
    # display('unknown', unknown)
    # #
    # # Marker labelling
    # ret, markers = cv.connectedComponents(sure_fg)
    # # Add one to all labels so that sure background is not 0, but 1
    # markers = markers + 1
    # # Now, mark the region of unknown with zero
    # markers[unknown == 255] = 0
    # display('unknown', unknown)
    #
    # ## watershed
    # markers = cv.watershed(img, markers)
    # img[markers == -1] = [255, 0, 0]

    # # contours
    # contours, hierarchy = cv.findContours(markers.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    # for i in range(len(contours)):
    #     if hierarchy[0][i][3] == -1:
    #         cv.drawContours(img, contours, i, (255, 0, 0), 1)
    #
    # display('sep_coins', img)



def main():
    img = cv.imread('C:/dev/Object-detection/testfiles_ara/0618/img/img_book_only.png')
    display('img', img)
    print(img.shape)
    watershed(img.copy())

if __name__ == "__main__":
    main()

