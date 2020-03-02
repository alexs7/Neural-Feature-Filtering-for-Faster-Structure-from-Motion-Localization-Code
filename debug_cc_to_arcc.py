from evaluator import show_projected_points_only_intrinsics
from evaluator import save_projected_points_only_intrinsics
import numpy as np
#
K = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/matrices/pixel_intrinsics_low_640_landscape.txt")
# Rt_points3D = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/points3D_AR.txt")
#
# show_projected_points_only_intrinsics("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/current_query_image/query.jpg", K, Rt_points3D)

from point3D_loader import get_points3D
from query_image import get_query_image_id_new_model
from evaluator import get_ARCore_pose_query_image
from query_image import get_query_image_global_pose_new_model
import numpy as np
import os

image_id_start = get_query_image_id_new_model("query.jpg")
points3D = get_points3D(image_id_start)
print("Number of COLMAP 3D Points: " + str(len(points3D)))

colmap_pose = get_query_image_global_pose_new_model("query.jpg")
print('COLMAP Pose')
print(colmap_pose)

arcore_pose = get_ARCore_pose_query_image()
print('ARCore Pose')
print(arcore_pose)

colmap_to_arcore_matrix = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]])
print('CM to ARC Matrix')
print(colmap_to_arcore_matrix)

rotZ = np.array([[0, 1, 0, 0],[-1, 0, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
print('Rotate -90 on Z (Anticlockwise)')
print(rotZ)

arcore_pose_inverse = np.linalg.inv(arcore_pose)

#from_colmap_world_to_colmap_camera
points3D = colmap_pose.dot(np.transpose(points3D))
#from_colmap_camera_to_arcore_camera
points3D = colmap_to_arcore_matrix.dot(rotZ.dot(points3D))
points3D = np.transpose(points3D)

save_projected_points_only_intrinsics("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/current_query_image/query.jpg", K, points3D)