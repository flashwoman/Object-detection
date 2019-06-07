# 모양을 구분해주는 ShapeDetector
import cv2 as cv


# # 전처리한 사진 다른이름으로 저장하기
# path = "C:/Users/DELL/PycharmProjects/Object-detection/image/test_shelf_resized.jpg"
# # resized_image.save(path)
# img = cv.imread(path, cv.IMREAD_COLOR)
# img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # shape : (600, 900)
#
#
# # cv2의 Corner Harris 함수 사용 (전처리 후)
# # 책장 검출 용이한 사진으로 전처리
# # ret,thresh1 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY)
# ret,thresh2 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY_INV) # 책 아웃라인 검출
#
#
# class ShapeDetector:
#
#     def __init__(self):
#
#         pass
#
#     def detect(self, c):
#
#         # initialize the shape name and approximate the contour
#         shape = "unidentified"
#         peri = cv.arcLength(c, True)
#         approx = cv.approxPolyDP(c, 0.04 * peri, True)
#
#
#         # if the shape is a triangle, it will have 3 vertices
#         if len(approx) == 3:
#
#             shape = "triangle"
#
#
#         # if the shape has 4 vertices, it is either a square or a rectangle
#         elif len(approx) == 4:
#
#             # compute the bounding box of the contour and use the
#             # bounding box to compute the aspect ratio
#             (x, y, w, h) = cv.boundingRect(approx)
#             ar = w / float(h)
#
#             # a square will have an aspect ratio that is approximately equal to one, otherwise, the shape is a rectangle
#             shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
#
#
#         # if the shape is a pentagon, it will have 5 vertices
#         elif len(approx) == 5:
#
#             shape = "pentagon"
#
#
#         # otherwise, we assume the shape is a circle
#         else:
#
#             shape = "circle"
#
#
#         # return the name of the shape
#         return shape
#
#
# # cv2.findContours 함수는 윤곽이 있는 부분을 찾아주는 함수이다.
#
# sd = ShapeDetector()
# cnts = cv.findContours(thresh2.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if imutils.is_cv2() else cnts[1] # 찾은 contour들 cnts에 저장된다.
#
# # 찾아진 contour들 사진에 표시해서 띄우기
# for c in cnts:
#
#     if sd.detect(c) != 'rectangle': next
#     c = c.astype("float")
#     c = c.astype("int")
#     x, y, w, h = cv.boundingRect(c)
#     cv.rectangle(thresh2, (x, y), (x + w, y + h), (3, 255, 4), 2)
#     cv.imshow("image", img)
#     cv.waitKey(0)
#
# cv.destroyAllWindows()




# 위 코드가 되지 않아서 찾아본 Reference Code <https://webnautes.tistory.com/1296>

def setLabel(image, str, contour):
    (text_width, text_height), baseline = cv.getTextSize(str, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    x,y,width,height = cv.boundingRect(contour)
    pt_x = x+int((width-text_width)/2)
    pt_y = y+int((height + text_height)/2)
    cv.rectangle(image, (pt_x, pt_y+baseline), (pt_x+text_width, pt_y-text_height), (200,200,200), cv.FILLED)
    cv.putText(image, str, (pt_x, pt_y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, 8)


# 이미지 불러오기
path = "C:/Users/DELL/PycharmProjects/Object-detection/image/test_shelf_resized.jpg"
img = cv.imread(path, cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 이미지 이진화시키기
ret,img_binary = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU) # 주의 : 배경이 검은색이고 검출도형이 하얀색이여야한다.
cv.imshow('result', img_binary)
cv.waitKey(0)


# 컨투어 검출하기
contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# RETR_EXTERNAL : 외곽의 컨투어만 검출하도록 한다. / CHAIN_APPROX_SIMPLE : 검출되는 컨투어의 구성점 개수 줄이기(ex. 직선부분 양끝점만 검출하기)


# 검출된 컨투어를 직선으로 근사화시키기
for cnt in contours:

    # 컨투어 구성성분 개수 확인하기
    size = len(cnt)
    print(size)

    # 컨투어를 직선으로 근사화시키기 2
    epsilon = 0.005 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)

    # 직선으로 근사화시킨 컨투어의 구성성분 개수 확인하기
    size = len(approx)
    print(size)
#
#     cv.line(img_color, tuple(approx[0][0]), tuple(approx[size-1][0]), (0, 255, 0), 3)
#     for k in range(size-1):
#         cv.line(img_color, tuple(approx[k][0]), tuple(approx[k+1][0]), (0, 255, 0), 3)
#
#     if cv.isContourConvex(approx):
#         if size == 3:
#             setLabel(img_color, "triangle", cnt)
#         elif size == 4:
#             setLabel(img_color, "rectangle", cnt)
#         elif size == 5:
#             setLabel(img_color, "pentagon", cnt)
#         elif size == 6:
#             setLabel(img_color, "hexagon", cnt)
#         elif size == 8:
#             setLabel(img_color, "octagon", cnt)
#         elif size == 10:
#             setLabel(img_color, "decagon", cnt)
#         else:
#             setLabel(img_color, str(size), cnt)
#     else:
#         setLabel(img_color, str(size), cnt)
#
# cv.imshow('result', img_color)
# cv.waitKey(0)