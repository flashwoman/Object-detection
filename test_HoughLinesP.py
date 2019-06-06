import numpy as np
import cv2

img = cv2.imread("C:/fleshwoman/Object-detection/image/bookshelf.jpg")
img2 = img.copy()
img_original = img.copy()
#1. GrayScale로 바꾸기
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)


#2. Edge 검출
edges = cv2.Canny(img, 5, 250)

cv2.imshow("edges", edges)
cv2.waitKey(0)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Closed", closed)
cv2.waitKey(0)


#



minLineLength = 100
maxLineGap = 10
rho = 1
theta = np.pi/180 * 90
threshold = 80
lines = cv2.HoughLinesP(edges, 1, theta, threshold, minLineLength, maxLineGap)
lines2 = cv2.HoughLinesP(closed, 1, theta, threshold, minLineLength, maxLineGap)

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
for i in range(len(lines)):
    for x1,y1,x2,y2 in lines[i]:
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

for i in range(len(lines2)):
    for x1,y1,x2,y2 in lines2[i]:
        cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 3)


images_combined = np.vstack((img, img2))
# images_combined_resized = cv.resize(images_combined, (400, -1))
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', images_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()


