import numpy as np
import csv
import matplotlib.pyplot as plt 
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = ap.parse_args()

# read roi data
cnts_info = []
with open(args.image, newline="") as csvFile:
    rows = csv.reader(csvFile)

    for i, r in enumerate(rows):
        if i != 0:
            r = [np.float(s) for s in r]
            cnts_info.append(r)

# caculate distance matric for each object
cnts_info = np.array(cnts_info)
ln = len(cnts_info)
cnts_dist = np.zeros((ln, ln))
for i, Ci in enumerate(cnts_info):
    for j, Cj in enumerate(cnts_info):
        if i > j:
            cnts_dist[i, j] = np.linalg.norm(Ci-Cj)
cnts_dist = cnts_dist + cnts_dist.T
np.fill_diagonal(cnts_dist, float("Inf"))

# find min distance for each object
min_dist = np.min(cnts_dist, axis=1)
np.fill_diagonal(cnts_dist, 0)

# show plot
# bins = np.linspace(min(min_dist), max(min_dist), max(min_dist) / 1)
plt.hist(min_dist, bins=np.arange(0, max(min_dist)))
plt.draw()
plt.waitforbuttonpress(0) 

dist_list = [i for i in cnts_dist.ravel() if i != 0]
bins = np.linspace(min(dist_list), max(dist_list), max(dist_list) / 10)
plt.hist(dist_list, bins=bins)
plt.draw()
plt.waitforbuttonpress(0)
plt.clf() 

w = np.max(cnts_info[:, 1]) - np.min(cnts_info[: 1])
h = np.max(cnts_info[:, 0]) - np.min(cnts_info[:, 0])
w_f = 6.5
h_f = round(h * 6/w, 1)
plt.figure(figsize=(h_f, w_f))
plt.scatter(cnts_info[:, 0], cnts_info[:, 1], s = cnts_info[:, 2] + 1)
plt.draw()
plt.waitforbuttonpress(0) 
plt.clf()
uniformity = np.var(min_dist)/np.mean(min_dist)
print("Coordinate CV:{}".format(uniformity))