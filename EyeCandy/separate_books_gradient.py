import numpy as np
import cv2 as cv
import glob, os


#########

# 1. Gradient로 정확한 더미를 찾고 그 이후에 커널 적용

##########

def display(winname, img):
    cv.moveWindow(winname, 500, 0)
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.imshow(winname, img)
    cv.waitKey(0)

def deletefiles(path, extension):
    files = f'*.{extension}'  # 찾고 싶은 확장자
    for file in glob.glob(os.path.join(path, files)):
        os.remove(file)


def preprocessing(img, h=2, w=2):
    kernel = np.ones((h, w), np.uint8)


    #MORPH_GRADIENT
    gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel ,iterations=2)
    display('gradient_h', gradient)

    v_gray = cv.cvtColor(gradient, cv.COLOR_BGR2GRAY)
    display('v_gray', v_gray)


    ret, img_binary = cv.threshold(v_gray, 50, 255, cv.THRESH_BINARY)
    display('img_binary', img_binary)
    return img_binary




def contours2(img, img_color):
    img_for_save = img_color.copy()
    ret, img_binary = cv.threshold(img, 127, 255, 0)
    path = "C:/flashwoman/Object-detection/EyeCandy/books/"
    deletefiles(path, 'jpg')  # 기존 파일 삭제

    boxes = []
    moments = []
    rects = []

    image, contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Rotated Rectangle
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        M = cv.moments(cnt)

        rects.append(rect)
        boxes.append(box.astype('int'))
        moments.append(M["m00"])


    mean = np.mean(moments)
    median = np.median(moments)
    rect = []
    for i, cnt in enumerate(contours):
        if (hierarchy[0][i][3] != -1) & (moments[i] > mean) & (moments[i] < mean * 2.5):
            cv.drawContours(img_color, [boxes[i]], -1, (3, 255, 4), 1)  # blue

            (x, y) = rects[i][0]
            (w, h) = rects[i][1]

            x, y, w, h = list(map(int, [x, y, w, h]))

            ## 2. 각각의 책 이미지 저장 ##
            img_result = img_for_save[y: y + h, x: x + w]
            img_path = f"{path}/img{i}.jpg"
            cv.imwrite(img_path, img_result)

    display('img_color', img_color)
    return img, rect


def main():
    img = cv.imread('C:/flashwoman/Object-detection/EyeCandy/img/img_book_only.png')
    img_h = img.copy()
    img_org = cv.imread('C:/flashwoman/Object-detection/EyeCandy/img/bookshelf_04.jpg')
    img_org_h = img_org.copy()

    # 전처리
    img = preprocessing(img_org)
    img, rect = contours2(img, img_org)


if __name__ == "__main__":
    main()

