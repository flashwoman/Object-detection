[TOC]

# HSV

**Goal : HSV 범위 자동화하기**

[Reference Page :  색상추출](# Reference Page : https://webnautes.tistory.com/1246)

[Reference Page : HSV 범위설정에 관하여](<https://bradbury.tistory.com/64>)



### Status

**책장색상 범위 가져오기**

1. Trackbar의 현재값 가져오기

   ```python
   threshold = cv.getTrackbarPos('threshold', 'img_color_morphed')
   ```

2. hsv 설정 Logic

   - Hue (색상) : Red와 non-Red로 나눈다 (0~180)

   - Saturaion (채도) : (흰)0~255, 선명도, 하얀색 범위 일부 제외하기 __ 범위 변경 필요

   - Value (명도) : (검)0~255(흰), 그림자를 검출하기위해 동일색상안의 모든 명도를 검출한다(단, 검은색과 하얀색 범위 일부 제외하기) __ 변경하지 않아도 된다.

   - 20190618 설정

     ```python
             else: # 10 < hsv[0] < 170 : 빨강이 아닌 나머지 컬러
                 print("case3")
                 lower_color1 = np.array([hsv[0], threshold, threshold])
                 upper_color1 = np.array([hsv[0] + 5, 40, 255])
                 lower_color2 = np.array([hsv[0] - 5, threshold, threshold])
                 upper_color2 = np.array([hsv[0], 40, 255])
                 lower_color3 = np.array([hsv[0] - 5, threshold, threshold])
                 upper_color3 = np.array([hsv[0], 40, 255])
     ```

3. 이미지 비트연산하기_책장을뺀 나머지영역 추출하기()

   ```python
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
       cv.imshow('mask', img_mask)
       img_inv_mask = cv.bitwise_not(img_mask)
       cv.imshow('inv_mask', img_inv_mask) # 반전마스크를 이용해 책만 추출하기
       
       # [sub] img_inv_mask로 책만 png로 추출하기
       img_book_only = cv.bitwise_and(img_color, img_color, mask=img_inv_mask)
       cv.imshow("img_book_only", img_book_only)
   ```

4. [배경 투명하게 만들기](<https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/>)

5. 