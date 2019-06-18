import numpy as np
import cv2 as cv
import imutils

def display(winname, img):
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.imshow(winname, img)
    cv.waitKey(0)



img_o = cv.imread('C:/fleshwoman/Object-detection/image/real_bookshelf_02.jpg')
img = cv.imread('b_and.jpg',cv.IMREAD_GRAYSCALE )
display('img',img)

# display('blur',blur)
# ret, thr2 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
# display('thr2',thr2)

# ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
# ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY)


class ShapeDetector:

    def __init__(self):
        pass

    def detect(self, c):

        # initialize the shape name and approximate the contour
        shape = "unidentified"
        epsilon = 0.04 * cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, epsilon, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape
