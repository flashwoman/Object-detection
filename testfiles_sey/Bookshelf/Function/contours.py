import cv2 as cv
from .display import display


if __name__ == '__main__':

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