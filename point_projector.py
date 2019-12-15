import numpy as np
import pdb
import cv2
import sys
import scipy.io as sio
import os
from images import get_query_image_pose
from images import get_query_image_id
from point3D_loader import get_points3D

image = cv2.imread("colmap_data/data6/current_query_image/query.jpg")
image_id = get_query_image_id()
K = np.loadtxt("matrices/pixel_intrinsics_low_640.txt")
Rt = get_query_image_pose()
points3D = get_points3D(image_id)

points = K.dot(Rt.dot(points3D.transpose())[0:3,:])
points = points // points[2,:]
points = points.transpose()

def project_points():
    for i in range(len(points)):
        x = int(points[i][0])
        y = int(points[i][1])
        center = (x, y)
        cv2.circle(image, center, 4, (0, 0, 255), -1)

    cv2.imwrite("result.png", image)

def get_projected_points():
    return points