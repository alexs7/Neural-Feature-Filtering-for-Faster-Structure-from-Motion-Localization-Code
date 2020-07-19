import glob
import os
import subprocess
import sys
import cv2
import numpy as np

# Before this you need the images in their own session folders
# dont forget trailing "/" if needed
# example: "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/official_datasets/CMU-Seasons-Extended/slice2/query/filtered/"
base_dir = sys.argv[1]

os.chdir(base_dir)
# from dataset README file fror camera 0
distortion_params = np.array([-0.399431, 0.188924, 0.000153, 0.000571])
fx = 868.993378
fy = 866.063001
cx = 525.942323
cy = 420.042529
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

for folder in glob.glob("session_*"):
    print("Doing Session: " + folder)
    for file in glob.glob(folder+"/"+"*.jpg"):
        img = cv2.imread(base_dir+file)
        h, w = img.shape[:2]
        dst = cv2.undistort(img, K, distortion_params, None, K)
        cv2.imwrite(base_dir+"undistorted/"+file, dst)

print("Done!")
