import cv2 as cv
import numpy as np


def nothing(x):
    pass


def mouse_callback(event, x, y, flags, param):

    global hsv, lower_color1, upper_color1, lower_color2, upper_color2, lower_color3, upper_color3

    # (마우스 왼쪽버튼으로) 선택한 위치의 픽셀값을 HSV로 변환합니다.
    if event == cv.EVENT_LBUTTONDOWN:
        print('x, y :', x, y)
        print('img_color[y, x]', img_color[y, x])
        color = img_color[y, x]
        one_pixel = np.uint8([[color]])
        hsv = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV)
        hsv = hsv[0][0]

        # ***** 사진에 따라 자동으로 반영되는 변수화 시키기 *****
        # Return the trackbar current position (Trackbar의 현재값 가져오기)
        threshold = cv.getTrackbarPos('threshold', 'detect_color') # default = -1

        # pixel 범위 설정하기
        if hsv[0] < 10:
            print("case1 : warm_red-ish")

            lower_color1 = np.array([ (hsv[0]-10+180), threshold, threshold ])
            upper_color1 = np.array([ 180, 255, 255 ])
            lower_color2 = np.array([ 0, threshold, threshold ])
            upper_color2 = np.array([ hsv[0], 255, 255 ])
            lower_color3 = np.array([ hsv[0], threshold, threshold ])
            upper_color3 = np.array([ (hsv[0]+10), 255, 255 ])

        elif hsv[0] > 170:
            print("case2 : cool_red-ish2")

            lower_color1 = np.array([hsv[0], threshold, threshold])
            upper_color1 = np.array([180, 255, 255])
            lower_color2 = np.array([0, threshold, threshold])
            upper_color2 = np.array([(hsv[0]+10-180), 255, 255])
            lower_color3 = np.array([hsv[0]-10, threshold, threshold])
            upper_color3 = np.array([(hsv[0] + 10), 255, 255])

        else:
            print("case3 : not red")

            lower_color1 = np.array([hsv[0], threshold, threshold])
            upper_color1 = np.array([(hsv[0]+5), 40, 255])
            lower_color2 = np.array([(hsv[0]-5), threshold, threshold])
            upper_color2 = np.array([hsv[0], 40, 255])
            lower_color3 = np.array([hsv[0]-5, threshold, threshold])
            upper_color3 = np.array([hsv[0], 40, 255])

        print("hsv:", hsv[0])
        print("@1", lower_color1, "~", upper_color1)
        print("@2", lower_color2, "~", upper_color2)
        print("@3", lower_color3, "~", upper_color3)


# 사용시 img_color 정의 필요