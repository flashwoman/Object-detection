#reference site : https://pythontips.com/2015/03/11/a-guide-to-finding-books-in-images-using-python-and-opencv/
# 배경 : canny -> houghLine 적용시 책표지 디자인에 따라 너무 많은 선이 만들어짐. test_HoughLines 에서 C:\fleshwoman\Object-detection\image\books.jpg 넣어서 참고

# import the necessary packages
import numpy as np
import cv2

# load the image, convert it to grayscale, and blur it
image = cv2.imread("C:/fleshwoman/Object-detection/image/books.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
# cv2.imshow("Gray", gray)
# cv2.waitKey(0)

# detect edges in the image
edged = cv2.Canny(gray, 5, 250)

# cv2.imshow("Edged", edged)
# cv2.waitKey(0)


# construct and apply a closing kernel to 'close' gaps between 'white'
# pixels
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closed", closed)
cv2.waitKey(0)

#######################################

# find contours (i.e. the 'outlines') in the image and initialize the
# total number of books found
# findContours(image, mode, method) : https://datascienceschool.net/view-notebook/f9f8983941254a34bf0fee42c66c5539/
(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0

# loop over the contours
for c in cnts: # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

# if the approximated contour has four points, then assume that the
# contour is a book -- a book is a rectangle and thus has four vertices
if len(approx) == 4:
    cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
    total += 1

# display the output
print(f'I found {total} books in that image')
cv2.imshow("Output", image)
cv2.waitKey(0)