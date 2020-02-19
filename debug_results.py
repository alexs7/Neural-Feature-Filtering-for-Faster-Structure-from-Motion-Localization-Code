from point3D_loader import get_points3D
from query_image import get_query_image_id_new_model
from evaluator import show_projected_points
from evaluator import get_ARCore_pose
from query_image import get_query_image_global_pose_new_model
import numpy as np

K = np.loadtxt("matrices/pixel_intrinsics_low_640.txt")

no = "1582042112257"

image_id_start = get_query_image_id_new_model("frame_"+no+".jpg")
points3D = get_points3D(image_id_start)

colmap_pose = get_query_image_global_pose_new_model("frame_"+no+".jpg")
arcore_pose = get_ARCore_pose("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_data/data_ar", no)
show_projected_points("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/current_query_image/frame_"+no+".jpg", K, colmap_pose, points3D)

points3D_AR = arcore_pose.dot(colmap_pose.dot(np.transpose(points3D)))
points3D_AR = np.transpose(points3D_AR)

np.savetxt('points3D_AR.txt',points3D_AR)

print("Done")
# maybe get the other ARCore poses here ?

# do the pose stuff here (colmap points to ARCore points ?)

# then write the points to a file