# 모양을 구분해주는 ShapeDetector
# [Reference Code](https://sosal.kr/1067)
import cv2 as cv
import imutils


# 전처리한 사진 다른이름으로 저장하기
path = "C:/Users/DELL/PycharmProjects/Object-detection/image/test_shelf_resized.jpg"
# resized_image.save(path)
img = cv.imread(path, cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # shape : (600, 900)


# cv2의 Corner Harris 함수 사용 (전처리 후)
# 책장 검출 용이한 사진으로 전처리
# ret,thresh1 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY_INV) # 책 아웃라인 검출


class ShapeDetector:

    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        epsilon = 0.005 * cv.arcLength(c, True) # 몇%나 직선으로 근사화시킬 것인지 결정 (현재는 0.5%)
        approx = cv.approxPolyDP(c, epsilon, True) # True이면 폐곡선 / False이면 열려있는 곡선

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately equal to one, otherwise, the shape is a rectangle
            if ar >= 0.95 and ar <= 1.05:
                shape = "square"

            else:
                "rectangle"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape


# cv2.findContours 함수는 윤곽이 있는 부분을 찾아주는 함수이다.
sd = ShapeDetector()
cnts = cv.findContours(thresh2.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1] # 찾은 contour들 cnts에 저장된다.
# print(cnts.shape) # (1, 513, 4)
print(cnts[0][0])
sd.detect(cnts[0][0])
# 찾아진 contour들 사진에 표시해서 띄우기
# for c in cnts:
    # print(c)
    # print(sd.detect(c))
    # if sd.detect(c) != 'rectangle':
    #     next

#     c = c.astype("float")
#     c = c.astype("int")
#     x, y, w, h = cv.boundingRect(c)
#     cv.rectangle(thresh2, (x, y), (x + w, y + h), (3, 255, 4), 2)
#     cv.imshow("image", img)
#     cv.waitKey(0)
#
# cv.destroyAllWindows()