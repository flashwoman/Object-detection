import cv2 as cv
import numpy as np

# 사진 불러오기(이 파일만 실행하는 경우만 실행시키기)
# # 받아온 사진 사이즈 고정시키기 (1100, 600)
# size = (900, 600)
# path = "C:/Users/DELL/PycharmProjects/Object-detection/image/bookshelf_04.jpg"
# img = Image.open(path)
# resized_image = img.resize(size)

# 전처리한 사진 다른이름으로 저장하기
path = "C:/Users/DELL/PycharmProjects/Object-detection/image/test_shelf_resized.jpg"
# resized_image.save(path)
img = cv.imread(path, cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # shape : (600, 900)


# cv2의 Corner Harris 함수 사용 (전처리 후)
# 책장 검출 용이한 사진으로 전처리
# ret,thresh1 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY_INV) # 책 아웃라인 검출
ret,thresh4 = cv.threshold(img_gray,127,255,cv.THRESH_TOZERO) # 책장 프레임 검출
thresh4_array = np.float32(thresh4)
thresh2_array = np.float32(thresh2)

thresh_bookshelf = thresh4_array + thresh2_array # bookshelf만을 선명하게 검출하기 위해

# print(img_dup)
# print(img_dup.shape) # (600, 900, 3)
dst = cv.cornerHarris(thresh_bookshelf, 5, 3, 0.04)
# print(dst)
print(dst.shape) # (600, 900) # 고정시킨 사진 size
print(dst.max())
#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.3*dst.max()]=[0,0,255]

cv.imshow('dst',img)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()

# corner point 찍힌 사진 저장하기
path = "C:/Users/DELL/PycharmProjects/Object-detection/image/images_thresh4_Point_func.jpg"
cv.imwrite(path, img)






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