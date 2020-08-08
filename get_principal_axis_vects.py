import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

data_file_path = sys.argv[1]
cameras_path = sys.argv[2]
path_to_text_file = sys.argv[3]

# These are the intrinsics from the db from the reference CMU model
fx = 1228.8
fy = 1228.8
cx = 512
cy = 384
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

with open(data_file_path) as f:
    images_data = f.readlines()

f = open(path_to_text_file, 'w')
for data in images_data:
    line = data.split(" ")

    name = line[0]
    qw = line[1]
    qx = line[2]
    qy = line[3]
    qz = line[4]

    tx = line[5]
    ty = line[6]
    tz = line[7]

    quat = [qx, qy, qz, qw]
    rot = R.from_quat(quat)
    rot = rot.as_dcm()

    M = K @ rot
    m3 = M[2, :]
    principal_axis_vector = np.linalg.det(M) * m3

    data = name + " " + str(qw) + " " + str(qx) + " " + str(qy) + " " + str(qz) + " " + str(tx) + " " + str(ty) + " " + str(tz) + \
           " " + str(principal_axis_vector[0]) + " " + str(principal_axis_vector[1]) + " " + str(principal_axis_vector[2]) + "\n"

    f.write(data)
f.close()
