import numpy as np
import cv2 as cv

img_o = cv.imread('C:/fleshwoman/Object-detection/image/real_bookshelf_02.jpg')
img1 = cv.imread('b_and.jpg')
img2 = cv.imread('b_or.jpg')

i_and = cv.bitwise_and(img1, img2)
cv.namedWindow('i_and', cv.WINDOW_NORMAL)
cv.imshow('i_and', i_and)

i_or = cv.bitwise_or(img1, img2)
cv.namedWindow('i_or', cv.WINDOW_NORMAL)
cv.imshow('i_or', i_or)


i_xor = cv.bitwise_xor(img1, img2)
cv.namedWindow('i_xor', cv.WINDOW_NORMAL)
cv.imshow('i_xor', i_xor)


img = cv.cvtColor(img_o, cv.COLOR_BGR2HSV)
cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.imshow('img', img)
print(img.shape)
cv.waitKey(0)

cv.namedWindow('img_h', cv.WINDOW_NORMAL)
cv.imshow('img_h', img[::1])
cv.waitKey(0)

cv.namedWindow('img_s', cv.WINDOW_NORMAL)
cv.imshow('img_s', img[::2])
cv.waitKey(0)

cv.namedWindow('img_v', cv.WINDOW_NORMAL)
cv.imshow('img_v', img[::3])
cv.waitKey(0)
