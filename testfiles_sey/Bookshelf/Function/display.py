import cv2 as cv

if __name__ == '__main__':

    def display(winname, img):

        cv.moveWindow(winname, 500, 0)
        cv.namedWindow(winname, cv.WINDOW_NORMAL)
        cv.imshow(winname, img)
        cv.waitKey(0)