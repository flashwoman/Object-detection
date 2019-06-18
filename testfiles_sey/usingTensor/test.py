# Reference Page : https://webnautes.tistory.com/1246

import cv2 as cv
import numpy as np

hsv = 0
lower_color1 = 0
upper_color1 = 0
lower_color2 = 0
upper_color2 = 0
lower_color3 = 0
upper_color3 = 0


# sv조절하기위해 트랙바 생성을 위한 공함수
def nothing(x):
    pass


def mouse_callback(event, x, y, flags, param):
    global hsv, lower_color1, upper_color1, lower_color2, upper_color2, lower_color3, upper_color3

    # 마우스 왼쪽 버튼 누를시 위치에 있는 픽셀값을 읽어와서 HSV로 변환합니다.
    if event == cv.EVENT_LBUTTONDOWN:
        print(img_color[y, x])
        color = img_color[y, x]

        one_pixel = np.uint8([[color]])
        hsv = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV)
        hsv = hsv[0][0]

        ###############################################
        ## 사진에 따라 자동반영되는 변수로 만들어줘야 할까? ##
        ###############################################
        threshold = cv.getTrackbarPos('threshold', 'img_result')  # Trackbar의 현재값을 가져와 threshold 변수에 넣어주기

        # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 필셀값의 범위를 정합니다.
        if hsv[0] < 10:
            print("case1")
            lower_color1 = np.array(
                [hsv[0] - 10 + 180, threshold, threshold])  # s, v가 낮을수록 검은색과 하얀색에 가까운 컬러가 많이 검출된다 (조명에 따라 조절해주기)
            upper_color1 = np.array([180, 255, 255])
            lower_color2 = np.array([0, threshold, threshold])
            upper_color2 = np.array([hsv[0], 255, 255])
            lower_color3 = np.array([hsv[0], threshold, threshold])
            upper_color3 = np.array([hsv[0] + 10, 255, 255])
            #     print(i-10+180, 180, 0, i)
            #     print(i, i+10)

        elif hsv[0] > 170:
            print("case2")
            lower_color1 = np.array([hsv[0], threshold, threshold])
            upper_color1 = np.array([180, 255, 255])
            lower_color2 = np.array([0, threshold, threshold])
            upper_color2 = np.array([hsv[0] + 10 - 180, 255, 255])
            lower_color3 = np.array([hsv[0] - 10, threshold, threshold])
            upper_color3 = np.array([hsv[0], 255, 255])
            #     print(i, 180, 0, i+10-180)
            #     print(i-10, i)
        else:
            print("case3")
            lower_color1 = np.array([hsv[0], threshold, threshold])
            upper_color1 = np.array([hsv[0] + 10, 255, 255])
            lower_color2 = np.array([hsv[0] - 10, threshold, threshold])
            upper_color2 = np.array([hsv[0], 255, 255])
            lower_color3 = np.array([hsv[0] - 10, threshold, threshold])
            upper_color3 = np.array([hsv[0], 255, 255])
            #     print(i, i+10)
            #     print(i-10, i)

        print(hsv[0])
        print("@1", lower_color1, "~", upper_color1)
        print("@2", lower_color2, "~", upper_color2)
        print("@3", lower_color3, "~", upper_color3)



path = "C:/Users/DELL/PycharmProjects/Object-detection/image/bookshelf_04.jpg"
img_color = cv.imread(path, cv.IMREAD_COLOR)
cv.imshow('origin', img_color)
cv.waitKey(0)

height, width = img_color.shape[:2]
img_color = cv.resize(img_color, (width, height), interpolation=cv.INTER_AREA)
# cv.imshow('resized', img_color)
# cv.waitKey(0)

# 원본 영상을 HSV 영상으로 변환합니다.
# img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)
cv.namedWindow('img_color')
cv.setMouseCallback('img_color', mouse_callback)

while(1):

    cv.imshow('img_color',img_color)

    k = cv.waitKey(1) & 0xFF

    if k == 27:
        print( "ESC 키 눌러짐")
        break

cv.destroyAllWindows()
