import numpy as np
import pdb
import cv2
import sys
import scipy.io as sio
import os
import point3D_loader

points3D = point3D_loader.parse_points3D("colmap_data/data5/new_model_text/points3D.txt", "57")
points3D = points3D.astype(np.float64)
points3D = np.hstack((points3D,np.ones((points3D.shape[0],1))))

image = cv2.imread("colmap_data/data5/query_images/query.jpg")
K = np.loadtxt("matrices/pixel_intrinsics_low_640.txt")

Rt = np.array([[ 0.9803, 0.1179, -0.1587, 1.3118],[-0.0436, 0.9119, 0.4081, 3.8028],[0.1929, -0.3931, 0.8991, -5.7931], [0, 0, 0, 1]])

temp = Rt.dot(points3D.transpose()).transpose()
temp = temp[:,0:3].transpose()
points = (K.dot(temp)).transpose()
points = points // points[:,2].reshape([878,1])
points = points[:,0:2]

for i in range(len(points)):
    x = int(points[i][0])
    y = int(points[i][1])
    center = (x, y)
    cv2.circle(image, center, 4, (0, 0, 255), -1)

cv2.imwrite("result.png", image)