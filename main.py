# Arguments
import cv2
from feature_matcher import FeatureMatcherTypes, feature_matcher_factory
from feature_matching_generator import feature_matcher_wrapper
from parameters import Parameters
from point3D_loader import read_points3d_default
import numpy as np

from ransac_comparison import run_comparison
from ransac_prosac import ransac, prosac

features_no = "1k" # colmap_features_no can be "2k", "1k", "0.5k", "0.25k"
exponential_decay_value = 0.5 # exponential_decay can be any of 0.1 to 0.9
db_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/database.db"
# images_path is the file of list of images to get 2D-3D matches from
# NOTE: Here you can use the "colmap_data/data/query_name.txt" if you want to exlude base images..
images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/images_localised_and_not_localised/" + features_no + "/images_localised.txt"
base_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/base_images.txt"
train_descriptors_all_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/"+features_no+"/points_mean_descs_all.npy"
train_descriptors_base_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/"+features_no+"/points_mean_descs_base.npy"

# by "complete model" I mean all the frames from future sessions localised in the base model (28/03)
complete_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/" + features_no + "/points3D.bin"
points3D = read_points3d_default(complete_model_points3D_path)  # base model's 3D points (same length as complete as we do not add points when localising new points, but different image_ds for each point)

# create points id and index relationship
point3D_index = 0
points3D_indexing = {}
for key, value in points3D.items():
    points3D_indexing[point3D_index] = value.id
    point3D_index = point3D_index + 1

# define matcher
matching_algo = FeatureMatcherTypes.FLANN  # or FeatureMatcherTypes.BF
match_ratio_test = Parameters.kFeatureMatchRatioTest
norm_type = cv2.NORM_L2
cross_check = False
matcher = feature_matcher_factory(norm_type, cross_check, match_ratio_test, matching_algo)

#distribution; row vector, same size as 3D points
points3D_avg_heatmap_vals = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/"+features_no+"/heatmap_matrix_avg_points_values_" + str(exponential_decay_value) + ".txt")
points3D_avg_heatmap_vals = points3D_avg_heatmap_vals.reshape([1, points3D_avg_heatmap_vals.shape[0]])

matches_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/"+features_no+"/matches_all.npy"

vanillia_data_path_save_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/vanilla_images_pose_"+str(exponential_decay_value)+".npy"
modified_data_path_save_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/modified_images_pose_"+str(exponential_decay_value)+".npy"

vanillia_data_path_save_info = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/vanillia_data_" + str(exponential_decay_value) + ".npy"
modified_data_path_save_info = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/modified_data_" + str(exponential_decay_value) + ".npy"

# Run Code here
# 1: Feature matching
matches = feature_matcher_wrapper(points3D_avg_heatmap_vals, db_path, images_path, train_descriptors_all_path, points3D, points3D_indexing, matcher)
breakpoint()

print("Saving matches...")
# save the 2D-3D matches
np.save(matches_path, matches)

# 2: RANSAC Comparison
vanilla_images_poses, vanilla_data, modified_images_poses, modified_data = run_comparison(0.5, ransac, prosac, matches_path, images_path, base_images_path)

print("Saving RANSAC data...")
# NOTE: folders .../RANSAC_results/"+features_no+"/... where created manually..