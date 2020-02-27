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

colmap_to_arcore_matrix = np.array([[1, 0, 0, 0],[0, -1, 0, 0],[0, 0, -1, 0],[0, 0, 0, 1]])
print('CM to ARC Matrix')
print(colmap_to_arcore_matrix)

rotate90Z = np.array([[0, -1, 0, 0],[1, 0, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
print('Rotate 90 on Z')
print(rotate90Z)

rotate90Z_clockwise = np.array([[0, -1, 0, 0],[1, 0, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
print('Rotate 90 on Z Clockwise')
print(rotate90Z_clockwise)

rotate90Z_anticlockwise = np.array([[0, 1, 0, 0],[-1, 0, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
print('Rotate 90 on Z Anticlockwise')
print(rotate90Z_anticlockwise)

arcore_pose_inverse = np.linalg.inv(arcore_pose)

from_colmap_world_to_colmap_camera = colmap_pose.dot(np.transpose(points3D))
from_colmap_camera_to_arcore_camera = colmap_to_arcore_matrix.dot(from_colmap_world_to_colmap_camera)
from_arcore_camera_to_arcore_world = arcore_pose_inverse.dot(from_colmap_camera_to_arcore_camera)
points3D_AR = np.transpose(from_arcore_camera_to_arcore_world)

# points3D_AR = arcore_pose_inverse.dot(rotate90Z_anticlockwise.dot(colmap_to_arcore_matrix.dot(colmap_pose.dot(np.transpose(points3D)))))
# points3D_AR = np.transpose(points3D_AR)

os.system("rm /Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/points3D_AR.txt")
np.savetxt('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/points3D_AR.txt', points3D_AR)