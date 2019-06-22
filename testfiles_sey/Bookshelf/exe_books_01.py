import cv2 as cv
import numpy as np
import unittest
import os
import sys
from .Function.display import display
from .Function.preprocessing import preprocessing
from .Function.contours import contours

sys.path.append(os.path.abspath("python_packaging/testfiles_sey/Bookshelf/Function"))


sys.path.append(os.path.abspath("/python_packaging/textedit/textedit/review"))
# use function list : display, preprocessing, cv.threshold

# load image
path = 'C:/dev/object-detection/testfiles_ara/0618/img/img_book_only.png'
img_color = cv.imread(path, cv.WINDOW_NORMAL)
display('img_color', img_color)
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
display('img_gray', img_gray)

# deal with no-background colors
ret, thresh = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
display('sep_threshold', thresh)
img_color = preprocessing(img_color)

# deal with similar-background colors
ret, thresh = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY_INV)
display('sep_threshold', thresh)
img_color = preprocessing(img_color)
img_color = contours(img_color)