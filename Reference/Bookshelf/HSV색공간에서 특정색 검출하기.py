# OpenCV : HSV = 0~ 255




# BGR to HSV 변환
import numpy as np
import cv2

color = [255, 0, 0] #Blue
pixel = np.unit8([[color]]) # cvtColor함수의 입력으로 사용하기 위해 : 한 픽셀로 구성된 이미지로 변환

hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV) # cvtColor 함수를 이용하여 HSV 색공간으로 변환
hsv = hsv[0][0] # hsv 값 출력을 위해 pixel 값만 가져오기

print('bgr: ', color)
print('hsv: ', hsv)

# Result
# bgr : [255, 0, 0]
# hsv: [120 255 255] # hue 값인 120 에서 +-10을 범위로 잡는다 (110~130)
# Saturation과 Value 값은 항상 일정한 범위값으로 사용하기 때문에 값을 무시한다.




# 이미지에서 파란색을 검출하는 코드
import cv2

img_color = cv2.imread('image.jpg') # color 검출할 이미지
height, width = img_color.shape[:2]

img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

# 범위를 정하여 hsv 이미지에서 원하는 색영역을 binary 이미지로 생성
lower_blue = (120-10, 30, 30) # sv : 너무 어두워서 검은색으로 보이는 영역과 너무 밝아 하얀색으로 보이는 영역을 제외시켜줌 (30, 30)
upper_blue = (120+10, 255, 255)
img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue) # 앞에서 정한 범위값을 사용하여 바이너리 이미지를 얻는다
    # 범위내의 픽셀들을 흰색이 되고, 나머지 픽셀들을 검은색으로 추출된다.

img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask) # 바이너리이미지를 마스크로 사용하여 범위값의 컬러를 선택

cv2.imshow('img_color', img_color)
cv2.imshow('img_mask', img_mask)
cv2.imshow('img_result', img_result)

cv2.waitKey(0)
cv2.destroyAllWindows()




# BGR to HSV 알고리즘 이용시 단점
# 컬러의 경계값을 인식시킬 때 두가지 색이 같은 범위에 포함될 수 있다. (좁은 범위를 사용하기...)
# 조명 변화에 따라 색이 다르게 보이는 것도 인식시키기 힘들다.