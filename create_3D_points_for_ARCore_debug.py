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

ar_core_inverse = np.linalg.inv(arcore_pose)

points3D_AR = ar_core_inverse.dot(rotate90Z.dot(colmap_to_arcore_matrix.dot(colmap_pose.dot(np.transpose(points3D)))))
points3D_AR = np.transpose(points3D_AR)

os.system("rm /Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/points3D_AR.txt")
np.savetxt('points3D_AR.txt', points3D_AR)