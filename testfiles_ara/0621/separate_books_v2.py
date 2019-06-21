import numpy as np
import cv2 as cv
import glob, os

###########################
# v2의 주제 = 여러가지 커널 사용
# 1. [ ㅡ , | , /, \] 네가지 방향의 커널 사용해보기
# 2. gradient 커널 사용해보기
#
#
###########################
# erode : 침식. open은 노이즈 없애기위해 erode -> delite
# delite와 close는 그 반대

###########################
# [ houghlineP ]
# 더미가 나뉘어져있으니까 houghlineP를 사용해도 되지 않을까?
#  더미 하나씩 작업

# [  ]
#
#
###########################

def display(winname, img):
    cv.moveWindow(winname, 500, 0)
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.imshow(winname, img)
    cv.waitKey(0)


def preprocessing(img):
    img_org = cv.imread('C:/dev/Object-detection/EyeCandy/img/bookshelf_04.jpg')
    ##### 1. vertical (잘세워진 책들 검출)
    # 1. kernel
    v_kernel = np.ones((3, 3), np.uint8)
    #morphologyEx
    v_opening = cv.morphologyEx(img_org.copy(), cv.MORPH_GRADIENT, v_kernel, iterations=2)

    # contours
    hierarchy, contours, hierarchy = cv.findContours(v_opening, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv.drawContours(img, contours, i, (255, 0, 0), 1)

    display('sep_coins', img)
    #
    # # 2. 그레이 스케일
    # v_gray = cv.cvtColor(v_opening, cv.COLOR_BGR2GRAY)
    #
    # # 3. 책 사이 간격 벌리기
    # kernel = np.ones((2, 1), np.uint8)
    # v_erode  = cv.morphologyEx(v_gray, cv.MORPH_ERODE, kernel, iterations=2)
    # display('v_erode', v_erode)
    #
    # # 4. contour로 넓이검사 -> 너무 넓으면 다른 방향의 책. 0으로 치환하자.
    # ret, img_binary = cv.threshold(img, 0, 255, 0)
    # display('img_binary', img_binary)
    #
    # ##### 1. horizontal (누운 책들 검출)
    # h_kernel = np.ones((2, 11), np.uint8)
    # # morphologyEx
    # h_opening = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, h_kernel, iterations=2)
    #
    # # 2. 그레이 스케일
    # h_gray = cv.cvtColor(h_opening, cv.COLOR_BGR2GRAY)
    #
    # # 3. 책 사이 간격 벌리기
    # h_kernel = np.ones((1, 2), np.uint8)
    # h_erode  = cv.morphologyEx(h_gray, cv.MORPH_ERODE, h_kernel, iterations=2)
    # display('h_erode', h_erode)

    return img


def contours(img, img_color):

    img_for_save = img_color.copy()

    # 1/ Threshold
    ret, img_binary = cv.threshold(img, 127, 255, 0)
    display("img_binary", img_binary)

    # 2. Contour 찾기
    imgs,contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # 3. 기존에 저장된 이미지 파일 삭제
    path = "C:/dev/Object-detection/EyeCandy/books/"
    files = '*.jpg'  # 찾고 싶은 확장자
    for file in glob.glob(os.path.join(path, files)):
        os.remove(file)


    ## 4. countour 그리고 저장하기.
    count = 0
    rect = []
    for cnt in contours:
        ## 1.  사각형 만들어 좌표얻기.
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(img_color, (x, y), (x + w, y + h), (3, 255, 4), 1)     # contour 그리기
        rect.append([(x, y), (x + w, y + h)])                               # rect에 좌표 저장 [Left_Top, Right_Bottom]

        ## 2. 각각의 책 이미지 저장 ##
        img_result = img_for_save[y: y + h, x: x + w]
        img_path = f"C:/dev/Object-detection/EyeCandy/books/img{count}.jpg"
        cv.imwrite(img_path,img_result)

        count = count + 1

    display("result", img_color)
    return img, rect



def main():
    img = cv.imread('C:/dev/Object-detection/EyeCandy/img/img_book_only.png')
    img_org = cv.imread('C:/dev/Object-detection/EyeCandy/img/bookshelf_04.jpg')
    display('img',img)
    # 전처리
    img = preprocessing(img)

    # contours
    #img, rect = contours(img,img_org)
    #print(rect)


if __name__ == "__main__":
    main()

