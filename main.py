# Arguments
import cv2

from database import COLMAPDatabase
from feature_matching_generator import feature_matcher_wrapper
from parameters import Parameters
from point3D_loader import read_points3d_default, get_points3D_xyz
import numpy as np
from pose_evaluator import pose_evaluate
from pose_refinement import refine_poses
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_query_images_pose_from_images
from ransac_comparison import run_comparison, sort_matches
from ransac_prosac import ransac, ransac_dist, prosac

features_no = "1k" # colmap_features_no can be "2k", "1k", "0.5k", "0.25k"
exponential_decay_value = 0.5 # exponential_decay can be any of 0.1 to 0.9

db_train = COLMAPDatabase.connect(Parameters.live_db_path)
model_images = read_images_binary(Parameters.live_model_images_path)

db_gt = COLMAPDatabase.connect(Parameters.qt_db_path) #this can be used to get the query images descs and ground truth poses for later pose comparison
all_query_images = read_images_binary(Parameters.gt_model_images_path)
all_query_images_names = load_images_from_text_file(Parameters.query_images_path)
localised_query_images_names = get_localised_image_by_names(all_query_images_names, Parameters.gt_model_images_path)

query_images_names = localised_query_images_names[0:10]
query_images_ground_truth_poses = get_query_images_pose_from_images(query_images_names, all_query_images)

# by "live model" I mean all the frames from future sessions localised in the base model
points3D = read_points3d_default(Parameters.live_model_points3D_path) # live model's 3d points have more images ids than base
points3D_xyz = get_points3D_xyz(points3D)

# train_descriptors_base and train_descriptors_live are self explanatory
# train_descriptors must have the same length as the number of points3D
train_descriptors_base = np.load(Parameters.avg_descs_base_path).astype(np.float32)
train_descriptors_live = np.load(Parameters.avg_descs_live_path).astype(np.float32)

# you can check if it is a distribution by calling, .sum() if it is 1, then it is.
# This can be either heatmap_matrix_avg_points_values_0.5.npy or reliability_scores_0.5.npy
# TODO: Do I need to normalise them ?
points3D_scores_1 = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/" + features_no + "/heatmap_matrix_avg_points_values_" + str(exponential_decay_value) + ".npy")
points3D_scores_2 = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/" + features_no + "/reliability_scores_" + str(exponential_decay_value) + ".npy")
points3D_live_model_scores = [points3D_scores_1, points3D_scores_2]

# 1: Feature matching
print("Feature matching...")

# TIP: Remember we are focusing on the model (and its descs) here so the cases to test are:
# query images , train_descriptors from live model : will match base + query images descs to live_model avg descs -> (this can have multiple cases depending on the points3D score used)
# query images , train_descriptors from base model : will match base + query images descs images descs to base avg descs -> (can only be one case...)

# query descs against base model descs
matches_base = feature_matcher_wrapper(db_gt, query_images_names, train_descriptors_base, points3D_xyz, Parameters.ratio_test_val, verbose = True)
np.save(Parameters.matches_base_save_path, matches_base)
matches_live = feature_matcher_wrapper(db_gt, query_images_names, train_descriptors_live, points3D_xyz, Parameters.ratio_test_val, True, points3D_live_model_scores)
np.save(Parameters.matches_live_save_path, matches_live)

matches_base = np.load(Parameters.matches_base_save_path).item()
matches_live = np.load(Parameters.matches_live_save_path).item()

# Print options
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

# 2: RANSAC Comparison
results = {}

print()
print("Base Model")
print("RANSAC")
poses , data = run_comparison(ransac, matches_base, query_images_names, verbose = True)
poses_refined = refine_poses(poses, matches_base)
trans_errors, rot_errors = pose_evaluate(poses_refined, query_images_ground_truth_poses)
results["ransac_base"] = [poses, data, poses_refined, trans_errors, rot_errors]

print("PROSAC only lowe's ratio - (lowes_distance_inverse_ratio)")
poses , data = run_comparison(prosac, matches_base, query_images_names, verbose = True, val_idx= Parameters.lowes_distance_inverse_ratio_index)
poses_refined = refine_poses(poses, matches_base)
trans_errors, rot_errors = pose_evaluate(poses_refined, query_images_ground_truth_poses)
results["prosac_base"] = [poses, data, poses_refined, trans_errors, rot_errors]

# -----

print()
print("Live Model")
print("RANSAC") #No need to run RANSAC multiple times here as it is not using any of the points3D scores
poses , data = run_comparison(ransac, matches_live, query_images_names, verbose = True)
poses_refined = refine_poses(poses, matches_live)
trans_errors, rot_errors = pose_evaluate(poses_refined, query_images_ground_truth_poses)
results["ransac_live"] = [poses, data, poses_refined, trans_errors, rot_errors]

print("RANSAC + dist")
poses, data = run_comparison(ransac_dist, matches_live, query_images_names, verbose = True, val_idx= Parameters.use_ransac_dist)
poses_refined = refine_poses(poses, matches_live)
trans_errors, rot_errors = pose_evaluate(poses_refined, query_images_ground_truth_poses)
results[Parameters.use_ransac_dist] = [poses, data, poses_refined, trans_errors, rot_errors]
results["ransac_dist_live"] = [poses, data, poses_refined, trans_errors, rot_errors]

prosac_value_indices = [ Parameters.lowes_distance_inverse_ratio_index,
                         Parameters.heatmap_val_index,
                         Parameters.reliability_score_index,
                         Parameters.reliability_score_ratio_index,
                         Parameters.custom_score_index,
                         Parameters.higher_neighbour_score_index]

print("PROSAC versions")
for prosac_sort_val in prosac_value_indices:
    poses, data = run_comparison(prosac, matches_live, query_images_names, verbose = True, val_idx= prosac_sort_val)
    poses_refined = refine_poses(poses, matches_live)
    trans_errors, rot_errors = pose_evaluate(poses_refined, query_images_ground_truth_poses)
    results["procac_live_"+str(prosac_sort_val)] = [poses, data, poses_refined, trans_errors, rot_errors]

np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/1k/results.npy", results)

print("Done!")
