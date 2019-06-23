import numpy as np
import cv2 as cv
import glob, os

def roate_rectange(img, img_color):
    img_for_save = img_color.copy()
    image, contours, hierachy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    path = "C:/dev/Object-detection/EyeCandy/books/"

    # 기존 파일 삭제
    deletefiles(path, 'jpg')

    boxes = []
    moments = []
    rects =[]
    for cnt in contours:
        # Rotated Rectangle
        rect = cv.minAreaRect(cnt)
        rects.append(rect)

        box = cv.boxPoints(rect)
        boxes.append(box.astype('int'))

        M = cv.moments(cnt)
        moments.append(M["m00"])


    mean = np.mean(moments)
    median = np.median(moments)

    rect = []
    for i, cnt in enumerate(contours):
        # if (moments[i] > median * 0.8 ) & (moments[i] < median * 1.5):
            print([boxes[i]])
            img2 = cv.drawContours(img_color, [boxes[i]], -1, 7)  # blue

            (x,y) = rects[i][0]
            (w,h) = rects[i][1]

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            ## 2. 각각의 책 이미지 저장 ##
            img_result = img_for_save[y: y + h, x: x + w]
            img_path = f"{path}/img{i}.jpg"
            cv.imwrite(img_path, img_result)

