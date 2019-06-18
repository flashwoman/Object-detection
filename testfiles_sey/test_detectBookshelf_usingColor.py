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

# 마우스이벤트를 감지할 윈도우 생성
cv.namedWindow('img_color_morphed', cv.WINDOW_NORMAL)
# cv.createTrackbar('threshold', 'img_color_morphed', 0, 255, nothing)
# cv.setTrackbarPos('threshold', 'img_color_morphed', 50)

def mouse_callback(event, x, y, flags, param):
    global hsv, lower_color1, upper_color1, lower_color2, upper_color2, lower_color3, upper_color3

    # 마우스 왼쪽 버튼 누를시 위치에 있는 픽셀값을 읽어와서 HSV로 변환합니다.
    if event == cv.EVENT_LBUTTONDOWN:
        print('x, y :', y, x)
        print('img_color[y, x]', img_color[y, x])
        color = img_color[y, x]

        one_pixel = np.uint8([[color]])
        hsv = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV)
        print(hsv)
        hsv = hsv[0][0]
        print(hsv)

        ###############################################
        ## 사진에 따라 자동반영되는 변수로 만들어줘야 할까? ##
        ###############################################
        # Returns the trackbar position
        threshold = cv.getTrackbarPos('threshold', 'img_color_morphed')  # Trackbar의 현재값을 가져와 threshold 변수에 넣어주기
        # print(threshold) # -1로 지정되어있다.

        # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 픽셀값의 범위를 정합니다.
        if hsv[0] < 10: # hue가 따뜻한 빨강에 가까울때
            print("case1")
            # s, v가 낮을수록 검은색과 하얀색에 가까운 컬러가 많이 검출된다 (조명에 따라 조절해주기)
            lower_color1 = np.array([hsv[0] - 10 + 180, threshold, threshold])
            upper_color1 = np.array([180, 255, 255])
            lower_color2 = np.array([0, threshold, threshold])
            upper_color2 = np.array([hsv[0], 255, 255])
            lower_color3 = np.array([hsv[0], threshold, threshold])
            upper_color3 = np.array([hsv[0] + 10, 255, 255])

        elif hsv[0] > 170: # 차가운 빨강에 가까운 컬러
            print("case2")
            lower_color1 = np.array([hsv[0], threshold, threshold])
            upper_color1 = np.array([180, 255, 255])
            lower_color2 = np.array([0, threshold, threshold])
            upper_color2 = np.array([hsv[0] + 10 - 180, 255, 255])
            lower_color3 = np.array([hsv[0] - 10, threshold, threshold])
            upper_color3 = np.array([hsv[0], 255, 255])

        else: # 10 < hsv[0] < 170 : 빨강이 아닌 나머지 컬러
            print("case3")
            lower_color1 = np.array([hsv[0], threshold, threshold])
            upper_color1 = np.array([hsv[0] + 5, 40, 255])
            lower_color2 = np.array([hsv[0] - 5, threshold, threshold])
            upper_color2 = np.array([hsv[0], 40, 255])
            lower_color3 = np.array([hsv[0] - 5, threshold, threshold])
            upper_color3 = np.array([hsv[0], 40, 255])


        print("hsv : ", hsv[0])
        print("@1", lower_color1, "~", upper_color1)
        print("@2", lower_color2, "~", upper_color2)
        print("@3", lower_color3, "~", upper_color3)


# 이름이 img_color_morphed인 윈도우에서 마우스 이벤트가 발생하면 mouse_callback함수가 호출됩니다.
cv.setMouseCallback('img_color_morphed', mouse_callback)

# # img sv 범위 실시간 변경하는 창 띄우기
# cv.namedWindow('img_result')
# cv.createTrackbar('threshold', 'img_result', 0, 255, nothing)
# cv.setTrackbarPos('threshold', 'img_result', 30)

# Video 실시간 캡쳐를 원할 때
# cap = cv.VideoCapture(0) # Video 캡쳐 객체 생성


while (True):
    # ret, img_color = cap.read() # 이미지를 웹캠으로부터 캡쳐하도록 한다. 조명의 영향을 더욱 많이 받음
    path = "C:/fleshwoman/Object-detection/image/real_bookshelf_02.jpg"
    img_color = cv.imread(path, cv.IMREAD_COLOR)
    # cv.imshow('origin', img_color)
    # cv.waitKey(0)

    height, width = img_color.shape[:2]
    img_color = cv.resize(img_color, (width, height), interpolation=cv.INTER_AREA)
    # cv.imshow('resized', img_color)
    # cv.waitKey(0)

    # 원본 영상을 HSV 영상으로 변환합니다.
    img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)
    # cv.setMouseCallback('img_color', mouse_callback)

    # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
    img_mask1 = cv.inRange(img_hsv, lower_color1, upper_color1)
    img_mask2 = cv.inRange(img_hsv, lower_color2, upper_color2)
    img_mask3 = cv.inRange(img_hsv, lower_color3, upper_color3)
    img_mask = img_mask1 | img_mask2 | img_mask3

    # img_result의 노이즈 제거하기(모폴로지 연산 이용)
    # kernel = np.ones((3, 3), np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))

    img_mask = cv.morphologyEx(img_mask, cv.MORPH_OPEN, kernel)
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_CLOSE, kernel)
    cv.namedWindow('mask', cv.WINDOW_NORMAL)
    cv.imshow('mask', img_mask)
    img_inv_mask = cv.bitwise_not(img_mask)
    cv.namedWindow('inv_mask', cv.WINDOW_NORMAL)
    cv.imshow('inv_mask', img_inv_mask) # 반전마스크를 이용해 책만 추출하기
    # 책장추출 마스크 저장하기
    path = "C:/fleshwoman/Object-detection/testfiles_ara/0618/img/img_book_mask.jpg"
    cv.imwrite(path, img_mask)

    # [sub] img_inv_mask로 책만 png로 추출하기
    img_book_only = cv.bitwise_and(img_color, img_color, mask=img_inv_mask)
    cv.namedWindow('img_book_only', cv.WINDOW_NORMAL)
    cv.imshow('img_book_only', img_book_only)
    path = "C:/fleshwoman/Object-detection/testfiles_ara/0618/img/img_book_only.jpg"
    cv.imwrite(path, img_book_only)

    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 책장 부분을 획득합니다.
    img_result = cv.bitwise_and(img_color, img_color, mask=img_mask)

    # 물체 위치 추적 코드(Labeling 필요 -> 중심좌표, 영역크기, 외곽박스 좌표를 얻을 수 있음)
    numOfLabels, img_label, stats, centroids = cv.connectedComponentsWithStats(img_mask)

    for idx, centroid in enumerate(centroids):
        if stats[idx][0] == 0 : #and stats[idx][0] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats[idx] # 사각영역
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > 50:  # 노이즈로 인해 검출된 작은 물체를 제거하기 위해 물체의 크기가 50 이상인 경우만 반환하기 (테스트하면서 조절필요)
            cv.circle(img_color, (centerX, centerY), 10, (0, 0, 255), 10)  # 중심점
            cv.rectangle(img_color, (x, y), (x + width, y + height), (0, 0, 255))  # 사각형으로 물체 잡아주기

    cv.imshow('img_color_morphed', img_result)
    path = "C:/fleshwoman/Object-detection/testfiles_ara/0618/img/img_color_morphed.jpg"
    cv.imwrite(path, img_result)

    # ESC 키누르면 종료
    if cv.waitKey(1) & 0xFF == 27:
        break



cv.destroyAllWindows()