import cv2 as cv
import numpy as np
import time


# 이미지 불러오기
path = "C:/Users/DELL/PycharmProjects/Object-detection/image/test_shelf_resized.jpg"
img = cv.imread(path, cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # shape : (600, 900)
ret,thresh4 = cv.threshold(img_gray,127,255,cv.THRESH_TOZERO) # 책장 프레임 검출



# Sobel 함수와 Harris Corner Detection 사용해서 thresh4(책장) 코너점 검출하기 (구현코드 --- 더 미세하게 조정할 수 있음)

img_sobel_x = cv.Sobel(thresh4, cv.CV_32F, 1, 0)
img_sobel_y = cv.Sobel(thresh4, cv.CV_32F, 0, 1)


IxIx = img_sobel_x * img_sobel_x
IyIy = img_sobel_y * img_sobel_y
IxIy = img_sobel_x * img_sobel_y


height, width = img.shape[:2]

window_size = 5
offset = int(window_size/2)

r = np.zeros(thresh4.shape)

start = time.process_time()
for y in range(offset, height-offset):
    for x in range(offset, width-offset):
        window_IxIx = IxIx[y-offset:y+offset+1, x-offset:x+offset+1]
        window_IyIy = IyIy[y-offset:y+offset+1, x-offset:x+offset+1]
        window_IxIy = IxIy[y-offset:y+offset+1, x-offset:x+offset+1]

        Mxx = window_IxIx.sum()
        Myy = window_IyIy.sum()
        Mxy = window_IxIy.sum()


        det = Mxx*Myy - Mxy*Mxy
        trace = Mxx + Myy

        r[y,x] = det - 0.04 * (trace ** 2)


cv.normalize(r,r,0.0,1.0,cv.NORM_MINMAX)

for y in range(offset, height-offset):
    for x in range(offset, width-offset):
        if r[y, x] > 0.4:
            img.itemset((y, x, 0), 0)
            img.itemset((y, x, 1), 0)
            img.itemset((y, x, 2), 255)


end = time.process_time()
print(end-start)

cv.imshow("original", img)
cv.waitKey(0)

# images_combiend 파일 저장하기
path = "C:/Users/DELL/PycharmProjects/Object-detection/image/images_thresh4_Point.jpg"
cv.imwrite(path, img)









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








# # cv2의 Corner Harris 함수 사용 (전처리 없이)
# path = "C:/Users/DELL/PycharmProjects/Object-detection/image/test_shelf_resized.jpg"
# img = cv.imread(path, cv.IMREAD_COLOR)
#
# img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# img_gray = np.float32(img_gray)
#
# # print(img_dup)
# # print(img_dup.shape) # (600, 900, 3)
# dst = cv.cornerHarris(img_gray, 5, 3, 0.04)
# # print(dst)
# print(dst.shape) # (600, 900) # 고정시킨 사진 size
# print(dst.max())
# #result is dilated for marking the corners, not important
# dst = cv.dilate(dst,None)
#
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.1*dst.max()]=[0,0,255] # Red Dot
#
# cv.imshow('dst',img)
#
# if cv.waitKey(0) & 0xff == 27:
#     cv.destroyAllWindows()
#
# # corner point 찍힌 사진 저장하기
# path = "C:/Users/DELL/PycharmProjects/Object-detection/image/images_origin_Point_func.jpg"
# cv.imwrite(path, img)