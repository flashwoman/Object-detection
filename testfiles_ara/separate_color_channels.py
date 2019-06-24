import numpy as np
import cv2 as cv
import glob, os
import imutils


def deletefiles(path, extension):
    files = f'*.{extension}'  # 찾고 싶은 확장자
    for file in glob.glob(os.path.join(path, files)):
        os.remove(file)

def display(winname, img):
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.imshow(winname, img)
    cv.waitKey(0)

def resize(img):
    ratio = img.shape[1] / 900.0
    img = imutils.resize(img, width=900)
    # height, width = img.shape[:2]  # [height, width, channel] = img.shape
    return img

##############################################################################################

#  'hsv'        : array([ 13,  20, 207]
# 'lower_color1': array([13, -1, -1]), 'upper_color1': array([ 18,  40, 255]), 
# 'lower_color2': array([ 8, -1, -1]), 'upper_color2': array([ 13,  40, 255]), 
# 'lower_color3': array([ 8, -1, -1]), 'upper_color3': array([ 13,  40, 255])


def canny(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, 10, 255)
    display('canny', canny)


def toHSV(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    display('hsv', hsv)

    # 빨간색만
    lower_red = np.array([-30, 30, 30])
    upper_red = np.array([50, 255, 255])

    # 파란색만
    lower_blue = np.array([110, 30, 30])
    upper_blue = np.array([179, 255, 255])

    # 초록색
    lower_green = np.array([50, 30, 30])
    upper_green = np.array([110, 255, 255])

    # 검은색
    lower_black = np.array([-30, 0, 0])
    upper_black = np.array([255, 30, 30])

    #mask
    mask_red = cv.inRange(hsv, lower_red, upper_red )
    mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv.inRange(hsv, lower_green , upper_green )
    mask_black = cv.inRange(hsv, lower_black, upper_black)

    # bitwise_and
    res_red = cv.bitwise_and(img, img, mask=mask_red)
    # display('res_red', res_red)
    res_blue = cv.bitwise_and(img, img, mask=mask_blue)
    # display('res_blue', res_blue)
    res_green = cv.bitwise_and(img, img, mask=mask_green)
    # display('res_green', res_green)
    res_black = cv.bitwise_or(img, img, mask=mask_black)
    # display('res_black', res_black)

    #threshold
    res_red = cv.cvtColor(res_red, cv.COLOR_BGR2GRAY)
    blur_red = cv.GaussianBlur(res_red, (5,5), 0)
    ret, thr_red = cv.threshold(blur_red, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    display('thr_red', thr_red)
    contours(thr_red, img)

    res_blue = cv.cvtColor(res_blue, cv.COLOR_BGR2GRAY)
    blur_blue = cv.GaussianBlur(res_blue, (5,5), 0)
    ret, thr_blue = cv.threshold(blur_blue, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # display('thr_blue', thr_blue)

    res_green = cv.cvtColor(res_green, cv.COLOR_BGR2GRAY)
    blur_green = cv.GaussianBlur(res_green, (5,5), 0)
    ret, thr_green= cv.threshold(blur_green, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # display('thr_green', thr_green)

    res_black = cv.cvtColor(res_black, cv.COLOR_BGR2GRAY)
    blur_black = cv.GaussianBlur(res_black, (5,5), 0)
    ret, thr_black= cv.threshold(blur_black, 30, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # display('thr_black', thr_black)
    fin = thr_red | thr_blue | thr_green | thr_black
    display('fin', fin)



    return fin


def contours(img, img_color):
    img_for_save = img_color.copy()
    image, contours, hierachy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    path = "C:/dev/Object-detection/EyeCandy/books/"

    # 기존 파일 삭제
    deletefiles(path, 'jpg')

    boxes = []
    moments = []
    rects =[]
    for cnt in contours:
        # Rotated Rectangle
        rect = cv.minAreaRect(cnt)
        rects.append(rect)

        box = cv.boxPoints(rect)
        boxes.append(box.astype('int'))

        M = cv.moments(cnt)
        moments.append(M["m00"])


    mean = np.mean(moments)
    median = np.median(moments)

    rect = []
    for i, cnt in enumerate(contours):
        # if (moments[i] > median * 0.8 ) & (moments[i] < median * 1.5):
            print([boxes[i]])
            img2 = cv.drawContours(img_color, [boxes[i]], -1 , (255,0,0),2)   # blue
            (x,y) = rects[i][0]
            (w,h) = rects[i][1]
            x,y,w,h = list(map(int, [x,y,w,h]) )
            ## 2. 각각의 책 이미지 저장 ##
            # img_result = img_for_save[y: y + h, x: x + w]
            # img_path = f"{path}/img{i}.jpg"
            # cv.imwrite(img_path, img_result)


def main():
    # img = cv.imread('C:/dev/Object-detection/image/img_book_only.jpg')
    # img_org = cv.imread('C:/dev/Object-detection/image/bookshelf_04.jpg')

    img = cv.imread('C:/dev/Object-detection/image/real_bookshelf_01.jpg')
    img_org = cv.imread('C:/dev/Object-detection/image/real_bookshelf_01.jpg')

    #resize
    resize(img)
    resize(img_org)
    display('img',img)

    mask = toHSV(img)

    hello = cv.bitwise_and(img, img, mask = mask)
    com = np.hstack([img_org, hello])
    display('com', com)



if __name__ == "__main__":

    main()
