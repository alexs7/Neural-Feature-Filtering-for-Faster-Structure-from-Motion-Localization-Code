import glob
import numpy as np
import cv2
from image_registrator import register_image
from query_image import get_query_image_global_pose
from point3D_loader import get_points3D
from query_image import get_query_image_id

K = np.loadtxt("matrices/pixel_intrinsics_low_640.txt")

def get_pose_from_correspondences(filename):
    correspondences = np.loadtxt(filename)
    (_, rvec, tvec, inliers_direct) = cv2.solvePnPRansac(correspondences[:,2:5], correspondences[:,0:2], K, None, iterationsCount = 500, confidence = 0.99, flags = cv2.SOLVEPNP_EPNP)
    rotM = cv2.Rodrigues(rvec)[0]
    pose = np.c_[rotM, tvec]
    pose = np.r_[pose, [np.array([0, 0, 0, 1])]]
    return pose

def show_projected_points(image_path, K, Pt, P0, GP, points3D):
    image = cv2.imread(image_path)
    points = K.dot(Pt.dot(P0).dot(GP.dot(points3D.transpose()))[0:3,:])
    points = points // points[2,:]
    points = points.transpose()
    for i in range(len(points)):
        x = int(points[i][0])
        y = int(points[i][1])
        center = (x, y)
        cv2.circle(image, center, 4, (0, 0, 255), -1)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow("result", image)
    cv2.waitKey(0)

# get global pose "GP" and local (t=0) P0 for query image
register_image("colmap_data/data/database.db", "colmap_data/data/current_query_image", "colmap_data/data/query_name.txt", "colmap_data/data/model/0", "colmap_data/data/new_model")
GP = get_query_image_global_pose()
P0 = get_pose_from_correspondences("colmap_data/data/current_query_image/correspondences.txt")

# get the 3D points the query image is looking at
image_id = get_query_image_id()
points3D = get_points3D(image_id)

# show projected points
show_projected_points("colmap_data/data/current_query_image/query.jpg", K, P0, np.linalg.inv(P0), GP, points3D)

# repeat for next frames
for fname in sorted(glob.glob("colmap_data/data/query_data/*.txt")):
    index_name = fname.split('_')[-1].split('.')[0]
    image_path = "colmap_data/data/query_data/frame_"+index_name+".jpg"
    Pt = get_pose_from_correspondences(fname)
    print(index_name)
    show_projected_points(image_path, K, Pt, np.linalg.inv(P0), GP, points3D)

