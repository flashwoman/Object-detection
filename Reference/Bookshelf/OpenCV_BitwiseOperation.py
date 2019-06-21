# Reference [http://blog.naver.com/PostView.nhn?blogId=pk3152&logNo=221442991872&parentCategoryNo=&categoryNo=46&viewDate=&isShowPopularPosts=true&from=search]

import cv2
import numpy as np

def imageBit():

    img1 = "image/logo.png"
    img2 = "image/coffee.jpg"

    # 이미지 지정
    im1 = cv2.imread(img1)
    im2 = cv2.imread(img2)

    # 로고의 행,열 크기값 받아오기
    rows, cols, channels = im1.shape
    # 삽입하고자 하는 이미지의 위치의 로고크기만큼의 이미지 컷팅
    roi = im2[10:10+rows , 10:10+cols]
    cv2.imshow("roi",roi)

    # 로고 이미지를 GRAYSCALE로 변환
    logoGray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    cv2.imshow("logoGray",logoGray)

    # 임계값 설정을 통한 mask 이미지 생성하기
    # threshold 라는 함수는 GRAYSCALE 만 사용가능함.
    # (대상이미지, 기준치, 적용값, 스타일)
    # 해당 cv2.THRESH_BINARY 는 이미지내의 픽셀값이 기준치 이상인 값들은
    # 모두 255로 부여함. 즉 픽셀값이 100이상이면 흰색, 100미만이면 검정색으로 표시
    # 변환된 이미지는 mask에 담김
    ret , mask = cv2.threshold(logoGray, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("mask",mask)

    # 임계값 설정을 한 이미지를 흑백 반전시킴
    mask_inv = cv2.bitwise_not(mask)
    cv2.imshow("mask_inv",mask_inv)

    # 위에서 자른 로고크기의 커피사진 영역에 mask에 할당된 이미지의
    # 0이 아닌 부분만 roi 와 roi 이미지를 AND 연산합니다.
    # 즉 커피이미지에서 로고크기만큼의 영역에 로고의 모양만 0값이 부여됩니다.
    im2_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    cv2.imshow("im2_bg",im2_bg)

    # 로고이미지에서 로고모양을 제외하고 다 0값을 가지게됩니다.
    im1_fg = cv2.bitwise_and(im1,im1,mask=mask)
    cv2.imshow("img_fg",im1_fg)

    # 로고크기만큼의 영역의 이미지에 로고이미지를 연산합니다.
    dst = cv2.add(im2_bg, im1_fg)
    cv2.imshow("dst",dst)

    # 커피의 원본이미지에 컷팅된 영역에 로고가 삽입된 이미지를 붙여넣습니다.
    im2[10:10+rows , 10:10+cols] = dst

    cv2.imshow('result', im2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


imageBit()

