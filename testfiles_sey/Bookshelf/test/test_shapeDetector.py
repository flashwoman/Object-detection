# Reference에 들어있는 OpneCV_ShapeDetect Code가 실행되지 않아 찾아본 다른 코드.
# Reference Code <https://webnautes.tistory.com/1296>
import cv2 as cv


def setLabel(image, str, contour):
    (text_width, text_height), baseline = cv.getTextSize(str, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    x,y,width,height = cv.boundingRect(contour)
    pt_x = x+int((width-text_width)/2)
    pt_y = y+int((height + text_height)/2)
    cv.rectangle(image, (pt_x, pt_y+baseline), (pt_x+text_width, pt_y-text_height), (200,200,200), cv.FILLED)
    cv.putText(image, str, (pt_x, pt_y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, 8)


# 이미지 불러오기
path = "C:/Users/DELL/PycharmProjects/Object-detection/image/test_shelf_resized.jpg"
img = cv.imread(path, cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 이미지 이진화시키기
ret,img_binary = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU) # 주의 : 배경이 검은색이고 검출도형이 하얀색이여야한다.
# ret,img_binary = cv.threshold(img_gray, 127, 255, cv.THRESH_TOZERO_INV|cv.THRESH_OTSU) # 주의 : 배경이 검은색이고 검출도형이 하얀색이여야한다.
cv.imshow('result', img_binary)
cv.waitKey(0)


# 컨투어 검출하기
contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# RETR_EXTERNAL : 외곽의 컨투어만 검출하도록 한다. / CHAIN_APPROX_SIMPLE : 검출되는 컨투어의 구성점 개수 줄이기(ex. 직선부분 양끝점만 검출하기)



# 검출된 컨투어를 직선으로 근사화시키기
for cnt in contours:

    # 컨투어 구성성분 개수 확인하기
    size1 = len(cnt)
    # print(size1)

    # 컨투어를 직선으로 근사화시키기 2
    epsilon = 0.001 * cv.arcLength(cnt, True) # 몇%나 직선으로 근사화시킬 것인지 결정 (현재는 0.5%)
    approx = cv.approxPolyDP(cnt, epsilon, True)

    # 직선으로 근사화시킨 컨투어의 구성성분 개수 확인하기
    size2 = len(approx)
    # print(size2-size1) # 대부분 0 or (-) 값이면 직선 근사화가 잘 된 것으로 파악한다.

    # 직선으로 근사화된 컨투어를 사진에 표시하기
    cv.line(img, tuple(approx[0][0]), tuple(approx[size2-1][0]), (0, 255, 0), 3) # 녹색선으로 표시
    for k in range(size2-1):
        cv.line(img, tuple(approx[k][0]), tuple(approx[k+1][0]), (0, 255, 0), 3) # 녹색선으로 표시

    # cv.imshow('result', img)
    # cv.waitKey(0)

    # 컨투어 내부에 도형 이름 표시하기
    if cv.isContourConvex(approx): # isContourConvex함수를 통해 오목하게 들어간 도형을 검출제외시킨다.
        if size2 == 3:
            setLabel(img, "triangle", cnt)
        elif size2 == 4:
            setLabel(img, "rectangle", cnt)
        elif size2 == 5:
            setLabel(img, "pentagon", cnt)
        elif size2 == 6:
            setLabel(img, "hexagon", cnt)
        elif size2 == 8:
            setLabel(img, "octagon", cnt)
        elif size2 == 10:
            setLabel(img, "decagon", cnt)
        else:
            setLabel(img, str(size2), cnt)
    else:
        setLabel(img, str(size2), cnt)


# img 파일 저장하기
path = "C:/Users/DELL/PycharmProjects/Object-detection/image/images_shapeDetected.jpg"
cv.imwrite(path, img)

cv.imshow('result', img)
cv.waitKey(0)
cv.destroyAllWindows()