import cv2 as cv
import numpy as np


# *** load image *** #
path = "C:/Object-detection/image/test_shelf_resized.jpg"
img_color = cv.imread(path, cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY) # shape : (600, 900)


# ***make binary image *** #
ret,thresh1 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY_INV) # to detect book
ret,thresh3 = cv.threshold(img_gray,127,255,cv.THRESH_TOZERO_INV)
ret,thresh4 = cv.threshold(img_gray,127,255,cv.THRESH_TOZERO) # to detect bookshelf
thresh2_array = np.float32(thresh2)
thresh4_array = np.float32(thresh4)
# thresh_bookshelf = cv.add(thresh2_array + thresh4_array)  # *** is it work better? ***


# *** detect corner *** #
thresh2_corner = cv.cornerHarris(thresh2_array, 5, 3, 0.04)
thresh2_corner = cv.dilate(thresh2_corner,None) # not important
img_color[thresh2_corner>0.5*thresh2_corner.max()]=[0,0,255] # Threshold : it may vary to the image.


# *** save image *** #
path = "C:/Object-detection/image/images_thresh4_Point_func.jpg"
cv.imwrite(path, img_color)


# *** show image *** #
cv.imshow('detect corner',img_color)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()

cv.destroyAllWindows()