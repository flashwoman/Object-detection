import cv2 as cv
import numpy as np


def trace_object(img_color, img_mask):

    global numOfLabels, img_label, stats, centroids

    centroids = cv.connectedComponentsWithStats(img_mask)

    for idx, centroid in enumerate(centroids):

        if stats[idx][0] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats[idx] # 사각영역
        centerX, centerY = int(centroid[0]), int(centroid[1])

        # Get rid of noise (noise : size under 50) / need to customize
        if area > 50 :
            cv.circle(img_color, (centerX, centerY), 10, (0,0,255), 10) # 중심점
            cv.rectangle(img_color, (x,y), (x+width,y+height), (0,0,255)) # 사각형으로 물체 잡기