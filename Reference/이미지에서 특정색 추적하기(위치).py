# Reference Page : https://webnautes.tistory.com/1246

import cv2 as cv
import numpy as np


hsv = 0
lower_blue1 = 0
upper_blue1 = 0
lower_blue2 = 0
upper_blue2 = 0
lower_blue3 = 0
upper_blue3 = 0


# sv조절하기위해 트랙바 생성을 위한 공함수
def nothing(x):
    pass

def mouse_callback(event, x, y, flags, param):
    global hsv, lower_blue1, upper_blue1, lower_blue2, upper_blue2, lower_blue3, upper_blue3

    # 마우스 왼쪽 버튼 누를시 위치에 있는 픽셀값을 읽어와서 HSV로 변환합니다.
    if event == cv.EVENT_LBUTTONDOWN:
        print(img_color[y, x])
        color = img_color[y, x]

        one_pixel = np.uint8([[color]])
        hsv = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV)
        hsv = hsv[0][0]
        
        threshold = cv.getTrackbarPos('threshold', 'img_result') # Trackbar의 현재값을 가져와 threshold 변수에 넣어주기

        # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 필셀값의 범위를 정합니다.
        if hsv[0] < 10:
            print("case1")
            lower_blue1 = np.array([hsv[0]-10+180, threshold, threshold]) # s, v가 낮을수록 검은색과 하얀색에 가까운 컬러가 많이 검출된다 (조명에 따라 조절해주기)
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, threshold, threshold])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0], threshold, threshold])
            upper_blue3 = np.array([hsv[0]+10, 255, 255])
            #     print(i-10+180, 180, 0, i)
            #     print(i, i+10)

        elif hsv[0] > 170:
            print("case2")
            lower_blue1 = np.array([hsv[0], threshold, threshold])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, threshold, threshold])
            upper_blue2 = np.array([hsv[0]+10-180, 255, 255])
            lower_blue3 = np.array([hsv[0]-10, threshold, threshold])
            upper_blue3 = np.array([hsv[0], 255, 255])
            #     print(i, 180, 0, i+10-180)
            #     print(i-10, i)
        else:
            print("case3")
            lower_blue1 = np.array([hsv[0], threshold, threshold])
            upper_blue1 = np.array([hsv[0]+10, 255, 255])
            lower_blue2 = np.array([hsv[0]-10, threshold, threshold])
            upper_blue2 = np.array([hsv[0], 255, 255])
            lower_blue3 = np.array([hsv[0]-10, threshold, threshold])
            upper_blue3 = np.array([hsv[0], 255, 255])
            #     print(i, i+10)
            #     print(i-10, i)

        print(hsv[0])
        print("@1", lower_blue1, "~", upper_blue1)
        print("@2", lower_blue2, "~", upper_blue2)
        print("@3", lower_blue3, "~", upper_blue3)


cv.namedWindow('img_color')
cv.setMouseCallback('img_color', mouse_callback)

# img sv 범위 실시간 변경하는 창 띄우기
# cv.namedWindow('img_result')
# cv.createTrackbar('threshold', 'img_result', 0, 255, nothing)
# cv.setTrackbarPos('threshold', 'img_result', 30)

# Video 실시간 캡쳐를 원할 때
# cap = cv.VideoCapture(0) # Video 캡쳐 객체 생성


while(True):
    # ret, img_color = cap.read() # 이미지를 웹캠으로부터 캡쳐하도록 한다. 조명의 영향을 더욱 많이 받음
    img_color = cv.imread('2.jpg') # 정적 이미지 이용
    height, width = img_color.shape[:2]
    img_color = cv.resize(img_color, (width, height), interpolation=cv.INTER_AREA)

    # 원본 영상을 HSV 영상으로 변환합니다.
    img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)

    # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
    img_mask1 = cv.inRange(img_hsv, lower_blue1, upper_blue1)
    img_mask2 = cv.inRange(img_hsv, lower_blue2, upper_blue2)
    img_mask3 = cv.inRange(img_hsv, lower_blue3, upper_blue3)
    img_mask = img_mask1 | img_mask2 | img_mask3
    
    
    # img_result의 노이즈 제거하기(모폴로지 연산 이용)
    kernel = np.ones((11, 11), np.unit8)
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_OPEN, kernel)
    img_mask = cv.morphologyEx(img_mask, cv_MORPH_CLOSE, kernel)


    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
    img_result = cv.bitwise_and(img_color, img_color, mask=img_mask)
    
    
    # 물체 위치 추적 코드(Labeling 필요 -> 중심좌표, 영역크기, 외곽박스 좌표를 얻을 수 있음)
    numOfLabels, img_label, stats, centroids = cv.connectedComponentsWithStats(img_mask)
    
    for idx, centroids in enumerate(centroids):
        if stats[idx][0] == 0 and stats[idx][0] == 0
            continue
        
        if np.any(np.isnan(centroids)):
            continue
            
        x, y, width, height, area = stats[idx]
        centerX, centerY = int(centroid[0]), int(centroid[1])
        
        if area > 50: # 노이즈로 인해 검출된 작은 물체를 제거하기 위해 물체의 크기가 50 이상인 경우만 반환하기 (테스트하면서 조절필요)
            cv.circle(img_color, (centerX, centerY), 10, (0,0,255), 10) # 중심점
            cv.rectangle(img_color, (x,y), (x+width, y+height), (0,0,255)) # 사각형으로 물체 잡아주기


    cv.imshow('img_color', img_color)
    cv.imshow('img_mask', img_mask)
    cv.imshow('img_result', img_result)


    # ESC 키누르면 종료
    if cv.waitKey(1) & 0xFF == 27:
        break


cv.destroyAllWindows()