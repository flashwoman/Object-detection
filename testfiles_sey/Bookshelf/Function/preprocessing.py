import cv2 as cv
import numpy as np
from .display import display

if __name__ == '__main__':
    def preprocessing(img_color, kernel_shape):

        # vertical kernel
        kernel = np.ones(kernel_shape, np.uint8)
        # morphologyEx
        kernel_opening = cv.morphologyEx(img_color, cv.MORPH_OPEN, kernel, iterations=2)

        # sure_bg = cv.dilate(opening, kernel, iterations=3)
        display('1st_opening', kernel_opening)

        gray = cv.cvtColor(kernel_opening, cv.COLOR_BGR2GRAY)
        display('gray', gray)

        ret, thresh = cv.threshold(gray, 10, 255, cv.THRESH_TOZERO)
        display('thresh', thresh)


        # horizontal kernel
        kernel_shape_T = kernel_shape.T
        kernel_shape_T = np.ones((3, 13), np.uint8)
        # morphologyEx
        kernel_opening_T = cv.morphologyEx(img_color, cv.MORPH_OPEN, kernel_shape_T, iterations=2)
        display('2nd_opening', kernel_opening_T)


        gray = cv.cvtColor(kernel_opening_T, cv.COLOR_BGR2GRAY)
        display('gray', gray)

        ret, thresh = cv.threshold(gray, 10, 255, cv.THRESH_TOZERO)
        display('thresh', thresh)

        bitwise_or = cv.bitwise_or(kernel_opening, kernel_opening_T )
        display('bitwise_or', bitwise_or)


        detected_books = cv.bitwise_and(img_color, img_color, mask=bitwise_or)
        display('bitwise_or', detected_books)

        return detected_books