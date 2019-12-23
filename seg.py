# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import cv2
import imutils
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

# load the input image and show its dimensions, keeping in mind that
# images are represented as a multi-dimensional NumPy array with
# shape no. rows (height) x no. columns (width) x no. channels (depth)
width = 450
resized = imutils.resize(image, width=width)
ratio = image.shape[1] / width
blurred = cv2.GaussianBlur(resized, (5, 5), 0)

# define the list of boundaries
boundary = ([20, 120, 110], [90, 200, 220])

# create NumPy arrays from the boundaries
lower = np.array(boundary[0], dtype = "uint8")
upper = np.array(boundary[1], dtype = "uint8")

# find the colors within the specified boundaries and apply
# the mask
mask = cv2.inRange(blurred, lower, upper)
output = cv2.bitwise_and(blurred, blurred, mask = mask)

gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)[1]
# edged = cv2.Canny(gray, 30, 150)

# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None
 
# ensure that at least one contour was found
if len(cnts) > 0:
	# sort the contours according to their size in
	# descending order
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	# loop over the sorted contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)

        # if our approximated contour has four points,
        # then we can assume we have found the paper
        if len(approx) == 4:
            docCnt = approx
            break


# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper

cv2.drawContours(resized, [docCnt], -1, [0, 255, 0], 2)
warped = four_point_transform(image, ratio*docCnt.reshape(4, 2))

cv2.imshow("Edged", warped)
cv2.waitKey(0)
