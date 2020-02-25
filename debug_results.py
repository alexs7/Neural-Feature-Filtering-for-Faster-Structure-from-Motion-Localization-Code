from point3D_loader import get_points3D
from query_image import get_query_image_id_new_model
from evaluator import show_projected_points
from evaluator import get_ARCore_pose_query_image
from query_image import get_query_image_global_pose_new_model
import numpy as np

K = np.loadtxt("matrices/pixel_intrinsics_low_640.txt")

image_id_start = get_query_image_id_new_model("query.jpg")
points3D = get_points3D(image_id_start)

# KEEP IN MIND the rotated image! - breaks! so you might need to rotate the image 90d anticlockwise
colmap_pose = get_query_image_global_pose_new_model("query.jpg")
show_projected_points("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/current_query_image/query.jpg", K, colmap_pose, points3D)