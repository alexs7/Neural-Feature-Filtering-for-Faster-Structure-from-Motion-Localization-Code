from evaluator import save_projected_points_only_intrinsics
from evaluator import get_ARCore_pose_query_image
import numpy as np

K = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/matrices/pixel_intrinsics_low_640_landscape.txt")

pointCloud = np.loadtxt('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/arcore_pointCloud.txt')
viewMatrix = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/arCoreViewMatrix.txt")
projMatrix = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/arCoreProjectionMatrix.txt")

arcore_pose = get_ARCore_pose_query_image()
print("arcore_pose: ")
print(arcore_pose)

arcore_pose_inverse = np.linalg.inv(arcore_pose)
print("arcore_pose_inverse: ")
print(arcore_pose_inverse)

points2D = projMatrix.dot(viewMatrix.dot(np.transpose(pointCloud)))
points2D = np.transpose(points2D)

breakpoint()

save_projected_points_only_intrinsics("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/current_query_image/query.jpg", K, points3D)