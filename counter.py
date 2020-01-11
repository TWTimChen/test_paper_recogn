# import the necessary packages
from imutils import contours
import cv2
import imutils
import argparse
import numpy as np

def pointTouchesImageBorder(point, imageHeight, imageWidth):
    
    pt = point
    retval = False

    xMin = 0
    yMin = 0
    xMax = imageWidth - 1
    yMax = imageHeight - 1
    # Use less/greater comparisons to potentially support contours outside of 
    # image coordinates, possible future workarounds with cv::copyMakeBorder where
    # contour coordinates may be shifted and just to be safe.
    if (pt[0] <= xMin or pt[1] <= yMin or pt[0] >= xMax or pt[1] >= yMax):
        retval = True

    return retval

def contourTouchesImageBorder(contour, imageHeight, imageWidth):

    cnt = contour
    retval = False

    touchBorder = [1 if pointTouchesImageBorder(p[0], imageHeight, imageWidth) else 0 for p in cnt]

    if sum(touchBorder) >= 1:
        retval = True

    return retval


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-s", "--show", action="store_true", help="show image for each steps")
ap.add_argument("-w", "--write", action="store_true", help="store spot_info to csv file")
args = ap.parse_args()

if args.show:
    show_list = []

image = cv2.imread(args.image)

# resize original image to assigned width
# and caculate resize ratio
height = 600
resized = imutils.resize(image, height=height)
width = resized.shape[1]
ratio = image.shape[1] / width
orig = resized.copy()
if args.show: 
    show_list.append(orig)

# transform color from BGR to LAB space
resized = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
if args.show: 
    show_list.append(resized)



Z = resized.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

if args.show:
    center = np.uint8(center)
    res = center[label.flatten()]
    res_orig = res.reshape((resized.shape))
    res_orig = cv2.cvtColor(res_orig, cv2.COLOR_LAB2BGR)
    show_list.append(res_orig)

# find the farest center
center_avg = np.sum(center, axis=0)
center_sst = (center - center_avg)**2
where_center_max = np.argmax(np.sum(center_sst, axis = 1))
center_alt = np.zeros_like(center)
center_alt[where_center_max] = center[where_center_max]

# Now convert back into uint8, and make original image
center_alt= np.uint8(center_alt)

res = center_alt[label.flatten()]
res_alt = res.reshape((resized.shape))
res_alt = cv2.cvtColor(res_alt, cv2.COLOR_BGR2GRAY)
if args.show: 
    show_list.append(res_alt)

# convert resized back to BGR
resized = cv2.cvtColor(resized, cv2.COLOR_LAB2BGR)

# find contours at res2
thresh = cv2.threshold(res_alt, 90, 255, cv2.THRESH_BINARY)[1]
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)

# remove contours which touchs image borders
cnts = [cnt for cnt in cnts if not contourTouchesImageBorder(cnt, height, width)]

# draw contours
cv2.drawContours(resized, cnts, -1, [0, 255, 0], 2)
if args.show: 
    show_list.append(resized)

spot_info = [["cX", "cY", "Area", "Peri"]]
# contour's center and area
for c in cnts:
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])
    else :
        cX, cY = np.mean(c.reshape(-1,2), axis=0, dtype=int)
    Area = cv2.contourArea(c)
    Peri = cv2.arcLength(c, True)
    spot_info.append([cX, cY, Area, round(Peri, 2)])

# write poi's x, y coordinate 
# and it's area and perimeter
if args.write:

    import csv

    image_name = (args.image).split(".")[1].split("/")[2]

    try :
        with open("./data/{}.csv".format(image_name), "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(spot_info)
            print("Success! \nWrite data into '{}'".format(csvfile.name))
    except:
        print("Fail to open path {}".format("./data/{}.csv".format(image_name)))


if args.show :
    for i, img in enumerate(show_list):
        cv2.imshow("step:{}".format(i), img)
        cv2.waitKey(0)
else :
    cv2.imshow('final step', np.hstack([resized, orig]))
    cv2.waitKey(0)

cv2.destroyAllWindows()
