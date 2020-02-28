from evaluator import show_projected_points_only_intrinsics
import numpy as np

K = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/matrices/pixel_intrinsics_low_640_portrait.txt")
Rt_points3D = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/points3D_AR.txt")

show_projected_points_only_intrinsics("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/current_query_image/query.jpg", K, Rt_points3D)
