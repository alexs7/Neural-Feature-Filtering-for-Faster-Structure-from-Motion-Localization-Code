# This file is for generating heatmaps of heatmap val, reliability socre and visibility score
# given a live image
import glob
import os
import sys

import numpy as np

from point3D_loader import read_points3d_binary_id_plus_xyz, read_points3d_binary, index_dict_reverse, \
    read_points3d_default
from query_image import read_images_binary, get_image_by_name, get_intrinsics_from_camera_bin, \
    get_query_image_pose_from_images, save_image_projected_points, save_heatmap_of_image

base_path = sys.argv[1] # i.e /home/alex/fullpipeline/colmap_data/CMU_data/slice3/

images = read_images_binary(base_path+"live/model/images.bin")
image_path = sys.argv[2] #/home/alex/fullpipeline/colmap_data/CMU_data/slice3/live/images/session_5/img_00870_c0_1287503885225127us.jpg
K = get_intrinsics_from_camera_bin(base_path+"live/model/cameras.bin", 2) # 2 since we are talking about live pics

# Note image might not be localised so you might need to look manually for one in images
name = image_path.split("/")[-2] + "/" + image_path.split("/")[-1]
image = get_image_by_name(name, images)

points3D_all = read_points3d_default(base_path+"live/model/points3D.bin")
points3D_image = read_points3d_binary_id_plus_xyz(base_path+"live/model/points3D.bin", image.id)
points3D_dict = index_dict_reverse(points3D_all)

query_pose = get_query_image_pose_from_images(name, images)
output_path = base_path + "/" + image_path.split("/")[-1] #might need to change that

# Getting the scores
points3D_reliability_scores_matrix= np.load(base_path+"reliability_scores.npy")
points3D_heatmap_vals_matrix = np.load(base_path+"heatmap_matrix_avg_points_values.npy")
points3D_visibility_matrix = np.load(base_path+"binary_visibility_values.npy")

points3D_reliability_scores = points3D_reliability_scores_matrix.sum(axis=0)
points3D_heatmap_vals = points3D_heatmap_vals_matrix.sum(axis=0)
points3D_visibility_vals = points3D_visibility_matrix.sum(axis=0)

points3D_reliability_scores = points3D_reliability_scores.reshape([1, points3D_reliability_scores.shape[0]])
points3D_heatmap_vals = points3D_heatmap_vals.reshape([1, points3D_heatmap_vals.shape[0]])
points3D_visibility_vals = points3D_visibility_vals.reshape([1, points3D_visibility_vals.shape[0]])

point3D_vals = []
for point3D in points3D_image:
    point3D_vals.append(points3D_reliability_scores[0, points3D_dict[point3D[3]]])

point3D_vals = np.array(point3D_vals)
save_heatmap_of_image(image_path, K, query_pose, points3D_image, output_path, point3D_vals)