import cv2 as cv
import numpy as np


def houghline(img_color):

    gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(img_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.namedWindow("houghline", cv.WINDOW_NORMAL)
    cv.imshow("houghline", img_color)
    cv.waitkey(0)
    cv.destroyAllWindows()


def houghline_p(img_color, minLineLength, maxLineGap):

    gray = cv.cvtColor(img_color,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150)
    lines = cv.HoughLinesP(edges,
                           1,
                           np.pi/180,
                           100,
                           minLineLength=minLineLength,
                           maxLineGap=maxLineGap)

    for line in lines:

        x1,y1,x2,y2 = line[0]
        cv.line(img_color, (x1,y1), (x2,y2), (0,255,0), 2)

    cv.namedWindow("houghline_p", cv.WINDOW_NORMAL)
    cv.imshow("houghline_p", img_color)
    cv.waitkey(0)
    cv.destroyAllWindows()

