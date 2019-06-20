# import numpy as np
# import cv2
#
# img = cv2.imread("C:/fleshwoman/Object-detection/image/test.jpg")
# img_original = img.copy()
# #1. GrayScale로 바꾸기
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #gray = cv2.GaussianBlur(gray, (3, 3), 0)
#
# cv2.imshow("gray", gray)
# cv2.waitKey(0)
#
#
# #2. Edge 검출
# edges = cv2.Canny(gray, 10, 250)
# cv2.imshow("edges", edges)
# cv2.waitKey(0)
#
#
# # https://m.blog.naver.com/samsjang/220505815055
# # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# # closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
# # closed = cv2.GaussianBlur(closed, (3, 3), 0)
# # #cv2.imshow("Closed", closed)
# # #cv2.waitKey(0)
# # #cv2.destroyAllWindows()
# #
#
# minLineLength = 100
# maxLineGap = 10
# rho = 1
# theta = np.pi / 5
# threshold = 350
# lines = cv2.HoughLines(closed, 1, theta, threshold)
# print(theta)
#
# for line in lines:
#     for rho,theta in line:
#      a = np.cos(theta)
#      b = np.sin(theta)
#      x0 = a*rho
#      y0 = b*rho
#      x1 = int(x0 + 1000*(-b))
#      y1 = int(y0 + 1000*(a))
#      x2 = int(x0 - 1000*(-b))
#      y2 = int(y0 - 1000*(a))
#      cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#      cropped = img[100:200, 500:640]
#
#
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# #
import cv2
import numpy as np

img = cv2.imread('C:/fleshwoman/Object-detection/image/test.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges1 = cv2.Canny(gray,50,150,apertureSize = 3)
# edges2 = cv2.Canny(gray,10,150,apertureSize = 3)
# edges3 = cv2.Canny(gray,50,300,apertureSize = 3) #No good 엄청 자잘함
# edges4 = cv2.Canny(gray,50,150,apertureSize = 5)


# images_row1 = np.hstack([edges1, edges2])
# images_row2 = np.hstack([edges3, edges4])
# images_combined = np.vstack((images_row1, images_row2))
#
# cv2.namedWindow('Images', cv2.WINDOW_NORMAL)
# cv2.imshow('Images', images_combined)
# cv2.waitKey(0)


lines = cv2.HoughLines(edges1,1,np.pi/180,200)
print(len(lines)) # 20本
for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow('output.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()