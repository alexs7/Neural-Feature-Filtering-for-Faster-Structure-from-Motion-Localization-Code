import numpy as np
import pdb
import cv2
import sys
import scipy.io as sio
import os


def project_points(image, points, index):

    image = cv2.imread(image)
    points = np.loadtxt(points)
    points = points[:,0:2]

    for i in range(len(points)):
        x = int(points[i][0])
        y = int(points[i][1])
        center = (x, y)
        cv2.circle(image, center, 4, (0, 0, 255), -1)

    cv2.imwrite("result_"+str(index)+".png", image)