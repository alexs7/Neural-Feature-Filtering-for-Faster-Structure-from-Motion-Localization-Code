from point3D_loader import get_points3D
from query_image import get_query_image_id_new_model
from evaluator import show_projected_points
from query_image import get_query_image_global_pose_new_model
import numpy as np

K = np.loadtxt("matrices/pixel_intrinsics_low_640.txt")

image_id_start = get_query_image_id_new_model("frame_1581881351213.jpg")
points3D = get_points3D(image_id_start)

colmap_pose = get_query_image_global_pose_new_model("frame_1581881351213.jpg")
show_projected_points("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/current_query_image/frame_1581881351213.jpg", K, colmap_pose, points3D)

breakpoint()
# maybe get the other ARCore poses here ?

# do the pose stuff here (colmap points to ARCore points ?)

# then write the points to a file