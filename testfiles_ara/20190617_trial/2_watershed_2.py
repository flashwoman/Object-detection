
import numpy as np
import cv2 as cv

# Median Blur
# GrayScale
# Binary ThresHold
# Find Contours

def display(winname, img):
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.imshow(winname, img)
    cv.waitKey(0)

# img_org = cv.imread('C:/fleshwoman/Object-detection/testfiles_ara\output/real_bookshelf_02_fin.jpg')
# # print(img_org.shape)
# sep_coins = cv.imread('C:/fleshwoman/Object-detection/testfiles_ara\output/real_bookshelf_02_fin.jpg')


img = cv.imread('b_or.jpg')
#img = cv.imread('C:/dev/Object-detection/0618/img/img_book_only.png')
#img = cv.imread('C:/fleshwoman/Object-detection/image/water_coins.jpg')

# Median Blur
sep_blur = cv.medianBlur(img, 15)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
display('gray',gray)



ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY +  cv.THRESH_OTSU)
display('sep_thr',thresh)

#NOISE REMOVAL (OPTIONAL)
kernel = np.ones((10,10), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
# kernel = np.ones((7,3), np.uint8)
# closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
# display('opening',opening)
# display('closing',closing)
#
# dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
# img_dist_transform = cv.normalize(dist_transform, None, 255, 0 , cv.NORM_MINMAX, cv.CV_8UC1)
# display('dist_transform',img_dist_transform)
#
# ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
# display('sure_fg',sure_fg)
#
# sure_fg = np.uint8(sure_fg)
# unknown = cv.subtract(sure_bg, sure_fg)
# display('unknown',unknown)
#
# # Marker labelling
# ret, markers = cv.connectedComponents(sure_fg)
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers + 1
# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0
# display('unknown',unknown)
#
#
# ## watershed
# markers = cv.watershed(img,markers)
# img[markers == -1] = [255,0,0]


# contours
contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

print(len(contours))
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv.drawContours(img, contours, i, (255,0,0), 1)

display( 'sep_coins', img)
