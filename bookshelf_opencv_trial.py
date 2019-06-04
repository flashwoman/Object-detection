# Trial points (image)
# - image resolution
# - image size
# - bookshelf shape(rounded, tilted, irregular, etc.)
# - bookshelf with books and other stuffs
# - include partial image of books
# - ratio : bookshelf - background

import cv2 as cv
import numpy as np
import imutils
import matplotlib.image as mpimg
from PIL import Image

# 받아온 사진 사이즈 고정시키기 (1100, 600)
size = (900, 600)
path = "C:/Users/DELL/PycharmProjects/Object-detection/image/bookshelf_04.jpg"
img = Image.open(path)
resized_image = img.resize(size)

# 전처리한 사진 다른이름으로 저장하기
path = "C:/Users/DELL/PycharmProjects/Object-detection/image/test_shelf_resized.jpg"
resized_image.save(path)

# 전처리된 사진 불러오기
img = cv.imread(path, cv.IMREAD_COLOR)
# print(img.shape) # (459, 612, 3)
cv.imshow("ImageShow", img)
cv.waitKey(0)

# 사진 -> GrayScale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # shape : (600, 900)
# img_gray_blurred = cv.GaussianBlur(img_gray, (5, 5), 0) # 가우시안블러 넣어주는게 좋을까?
cv.imshow("ImageShow", img_gray)
cv.waitKey(0)

# 5가지 threshold 방법
ret,thresh1 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img_gray,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img_gray,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img_gray,127,255,cv.THRESH_TOZERO_INV)

# resize each threshes
# img_gray = img_gray.reshape(200, -1)
# thresh1 = thresh1.reshape(200, -1)
# thresh2 = thresh2.reshape(200, -1)
# thresh3 = thresh3.reshape(200, -1)
# thresh4 = thresh4.reshape(200, -1)
# thresh5 = thresh5.reshape(200, -1)


# h-vstack으로 이미지 array 합쳐주기
images_row1 = np.hstack([img_gray, thresh1, thresh2])
images_row2 = np.hstack([thresh3, thresh4, thresh5])
# each row shape : (600, 2700) ===> resize to (400, 4050)
# images_row1 = images_row1.reshape(400, 1800)
# images_row1 = images_row1.reshape(400, -1)
# cv.imshow('Images', images_row1)
# cv.waitKey(0)
# print(images_row1.shape)
# print(images_row2.shape)


images_combined = np.vstack((images_row1, images_row2))

# images_combined_resized = cv.resize(images_combined, (400, -1))
cv.namedWindow('Images', cv.WINDOW_NORMAL)
cv.imshow('Images', images_combined)
cv.waitKey(0)
cv.destroyAllWindows()

# images_combiend 파일 저장하기
img = Image.open(path)
path = "C:/Users/DELL/PycharmProjects/Object-detection/image/images_combiend.jpg"
img.save(path)

# def draw_graph(x, y):
#     plt.plot(x, y, marker='o')
#     plt.xlabel('Distance in meters')
#     plt.ylabel('Gravitational force in newtons')
#     plt.title('Gravitational force and distance')
#
#     fig = plt.gcf() #변경한 곳
#     plt.show()
#     fig.savefig('GG.pdf') #변경한 곳

'''
# 사진 크롭하기
# 사진 크롭할 위치 찾는 코드 완성 후 주석 풀기
cropImage = im.crop((100, 100, 150, 150)) # crop()의 파라미터는 (좌, 상, 우, 하) 위치를 갖는 튜플로 지정한다.
cropImage.save('python-crop.jpg')
'''






# edges = cv.Canny(gray,50,200)
# lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=1,maxLineGap=10)
#
# for line in lines:
#     x1,y1,x2,y2 = line[0]
#     cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
#
#
# im = Image.open("test.jpg")
# im.save('test.jpg')
# cv.imshow("result", img)
# cv.waitKey(0)
