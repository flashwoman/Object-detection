# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-c", "--clusters", required=True, type=int, help="# of clusters")
args = vars(ap.parse_args())

# load the image and convert it from BGR to RGB so that
# we can display it with matplotlib
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BRG2RGB)

# show our image
plt.figure()
plt.axis("off")
plt.imshow(image)