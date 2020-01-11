# import the necessary packages
from imutils.perspective import four_point_transform
from scipy.spatial import distance as dist
from imutils import contours
import cv2
import imutils
import argparse
import numpy as np


def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
 
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
 
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
 
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis,:], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
 
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-s", "--show", action="store_true", help="show image for each steps")
ap.add_argument("-w", "--write", action="store_true", help="store the proccessed wrap image")
args = ap.parse_args()
image = cv2.imread(args.image)

if args.show:
    show_list = []

# load the input image and show its dimensions, keeping in mind that
# images are represented as a multi-dimensional NumPy array with
# shape no. rows (height) x no. columns (width) x no. channels (depth)
width = 450
resized = imutils.resize(image, width=width)
if args.show: 
    show_list.append(resized) 

ratio = image.shape[1] / width
# blurred = cv2.GaussianBlur(resized, (5, 5), 0)
blurred = cv2.medianBlur(resized, 15)
if args.show: 
    show_list.append(blurred)
 
# define the list of boundaries
boundary = ([20, 120, 110], [90, 200, 220])

# create NumPy arrays from the boundaries
lower = np.array(boundary[0], dtype = "uint8")
upper = np.array(boundary[1], dtype = "uint8")

# find the colors within the specified boundaries and apply
# the mask
mask = cv2.inRange(blurred, lower, upper)
output = cv2.bitwise_and(blurred, blurred, mask = mask)
if args.show: 
    show_list.append(output) 

# gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
if args.show: 
    show_list.append(gray) 

preCnts = cv2.Canny(gray, 30, 150)
# preCnts = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY)[1]

kernel = np.ones((10,10),np.uint8)
preCnts = cv2.dilate(preCnts, kernel = kernel, iterations = 1)
preCnts = cv2.erode(preCnts, kernel = kernel,iterations = 1)

if args.show: 
    show_list.append(preCnts)


# find contours in the thresholded image
cnts = cv2.findContours(preCnts.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
if args.show:
    resized_1 = resized.copy()
    cv2.drawContours(resized_1, cnts, -1, [0, 255, 0], 2)
    show_list.append(resized_1)

paperCnt = None
paperCntPoly = None
 
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
            paperCnt = approx
            paperCntPoly = c
            break


# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper
resized_2 = resized.copy()
cv2.drawContours(resized_2, [paperCnt], -1, [0, 255, 0], 2)

# warped = four_point_transform(image, ratio*paperCnt.reshape(4, 2))


# sort vertices clockwise by order_points 
# and create a new empty array for inscribed rectangle
paperCntInscribe = np.zeros_like(paperCnt)
paperCnt = order_points(paperCnt.reshape(4, 2))

# compare value for upper bound, lower bound,
# right bound and left bound respectively
paperCntInscribe[:2, :, 1] = np.max(paperCnt[:2, 1])
paperCntInscribe[2:, :, 1] = np.min(paperCnt[2:, 1])
paperCntInscribe[1:3, :, 0] = np.min(paperCnt[1:3, 0])
paperCntInscribe[[0, 3], :, 0] = np.max(paperCnt[[0, 3], 0])


cv2.drawContours(resized_2, [paperCntInscribe], -1, [0, 0, 255], 2)
warped = four_point_transform(image, ratio*paperCntInscribe.reshape(4, 2))

# show poccess loop
if args.show :
    show_list.append(resized_2)
    for i, img in enumerate(show_list):
        cv2.imshow("step:{}".format(i), img)
        cv2.waitKey(0)
else :
    cv2.imshow('final step', resized_2)
    cv2.waitKey(0)

cv2.imshow("ROI", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

if args.write:
    image_name = (args.image).split(".")[1].split("/")[2]
    cv2.imwrite("./img/{}_seg.jpg".format(image_name), warped)
