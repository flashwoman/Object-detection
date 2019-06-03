import numpy as np
import cv2

opened = cv2.imread("C:/Users/multicampus/fleshwoman/imgs/books.jpg")
minLineLength = 100
maxLineGap = 10
rho = 1
theta = np.pi/180
threshold = 190
lines = cv2.HoughLines(opened, 1, np.pi/180, threshold)

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
     cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
     cropped = image[100:200, 500:640]


print(lines, len(lines))
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)

# -*- coding:utf-8-*-
# import cv2
# import numpy as np
#
# img = cv2.imread("C:/Users/multicampus/fleshwoman/imgs/books.jpg")
# img_original = img.copy()
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,150,500,apertureSize=3)
# lines = cv2.HoughLines(edges,1,np.pi/180,100)
# for i in range(len(lines)):
#     for rho, theta in lines[i]:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + 1000*(-b))
#         y1 = int(y0+1000*(a))
#         x2 = int(x0 - 1000*(-b))
#         y2 = int(y0 -1000*(a))
#         cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#
# res = np.vstack((img_original,img))
# cv2.imshow('img',res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
# img = cv2.imread("C:/Users/multicampus/fleshwoman/imgs/bookshelf.jpg")
# edges = cv2.Canny(img,50,400,apertureSize = 3)
# gray = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
# minLineLength = 500
# maxLineGap = 4
#
# lines = cv2.HoughLinesP(edges,1,np.pi/360,100,minLineLength,maxLineGap)
# for i in range(len(lines)):
#     for x1,y1,x2,y2 in lines[i]:
#         cv2.line(img,(x1,y1),(x2,y2),(0,0,255),3)
#
#
# cv2.imshow('img1',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()