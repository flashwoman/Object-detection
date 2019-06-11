import numpy as np
import cv2 as cv

print(np.ones((11, 11), np.int8).dtype)
cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))

