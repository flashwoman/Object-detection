import numpy as np
import cv2 as cv
import glob, os




def display(winname, img):
    cv.moveWindow(winname, 500, 0)
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.imshow(winname, img)
    cv.waitKey(0)


def preprocessing(img):

    ## 1. 세로로 세워진 책들 색 균등화
    #. vertical kernel
    v_kernel = np.ones((11, 2), np.uint8)
    #morphologyEx
    v_opening = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, v_kernel, iterations=2)

    # 2. 그레이 스케일
    v_gray = cv.cvtColor(v_opening, cv.COLOR_BGR2GRAY)

    # 3. 책 사이 간격 벌리기
    kernel = np.ones((2, 1), np.uint8)
    erode  = cv.morphologyEx(v_gray, cv.MORPH_ERODE, kernel, iterations=2)
    display('erode', erode)

    return erode


def contours(img, img_color):

    img_for_save = img_color.copy()

    # 1/ Threshold
    ret, img_binary = cv.threshold(img, 127, 255, 0)
    display("img_binary", img_binary)

    # 2. Contour 찾기
    imgs,contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # 3. 기존에 저장된 이미지 파일 삭제
    path = "C:/flashwoman/Object-detection/EyeCandy/books/"
    files = '*.jpg'  # 찾고 싶은 확장자
    for file in glob.glob(os.path.join(path, files)):
        os.remove(file)


    ## 4. countour 그리고 저장하기.
    count = 0
    rect = []
    for cnt in contours:
        ## 1.  사각형 만들어 좌표얻기.
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(img_color, (x, y), (x + w, y + h), (3, 255, 4), 2)     # contour 그리기
        rect.append([(x, y), (x + w, y + h)])                               # rect에 좌표 저장 [Left_Top, Right_Bottom]

        ## 2. 각각의 책 이미지 저장 ##
        img_result = img_for_save[y: y + h, x: x + w]
        img_path = f"C:/flashwoman/Object-detection/EyeCandy/books/img{count}.jpg"
        cv.imwrite(img_path,img_result)

        count = count + 1

    display("result", img_color)
    return img, rect



def main():
    img = cv.imread('C:/flashwoman/Object-detection/EyeCandy/img/img_book_only.png')
    img_org = cv.imread('C:/flashwoman/Object-detection/EyeCandy/img/bookshelf_04.jpg')

    # 전처리
    img = preprocessing(img)

    # contours
    img, rect = contours(img,img_org)
    print(rect)



if __name__ == "__main__":
    main()

