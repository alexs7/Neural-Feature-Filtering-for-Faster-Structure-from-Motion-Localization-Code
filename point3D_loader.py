import sqlite3
import numpy as np
import sys
import pdb

def load_points3D(points3D_text_file):
    points3D = np.empty((0, 4))
    f = open(points3D_text_file, 'r')
    lines = f.readlines()
    lines = lines[3:] #skip comments
    f.close()

    for i in range(len(lines)):
        line = lines[i].split(" ")
        point3Did = line[0]
        point3Did_x = line[1]
        point3Did_y = line[2]
        point3Did_z = line[3]
        points3D = np.append(points3D, [point3Did, point3Did_x, point3Did_y, point3Did_z ])

    points3D = np.reshape(points3D, [int(np.shape(points3D)[0]/4),4])
    points3D = points3D.astype(float)
    return points3D