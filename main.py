# Arguments
import cv2

from database import COLMAPDatabase
from feature_matching_generator import feature_matcher_wrapper_live, feature_matcher_wrapper_base
from parameters import Parameters
from point3D_loader import read_points3d_default, get_points3D_xyz
import numpy as np
from pose_evaluator import pose_evaluate
from query_image import read_images_binary, load_images_from_text_file, get_images_names_bin, get_localised_image_by_names
from ransac_comparison import run_comparison, sort_matches
from ransac_prosac import ransac, ransac_dist, prosac

features_no = "1k" # colmap_features_no can be "2k", "1k", "0.5k", "0.25k"
exponential_decay_value = 0.5 # exponential_decay can be any of 0.1 to 0.9

db_train = COLMAPDatabase.connect(Parameters.live_db_path)
model_images = read_images_binary(Parameters.live_model_images_path)

db_gt = COLMAPDatabase.connect(Parameters.qt_db_path) #this can be used to get the query images descs and ground truth poses for later pose comparison
# qt_images_localised = read_images_binary(Parameters.gt_model_images_path)
all_query_images_names = load_images_from_text_file(Parameters.query_images_path)
localised_query_images_names = get_localised_image_by_names(all_query_images_names, Parameters.gt_model_images_path)

query_images_names = localised_query_images_names[0:10]

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

# 1: Feature matching
print("Feature matching...")

# TIP: Remember we are focusing on the model (and its descs) here so the cases to test are:
# query images , train_descriptors from live model : will match base + query images descs to live_model avg descs -> (this can have multiple cases depending on the points3D score used)
# query images , train_descriptors from base model : will match base + query images descs images descs to base avg descs -> (can only be one case...)

# Matches save locations
matches_1_save_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/" + features_no + "/matches_1.npy"
matches_2_save_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/" + features_no + "/matches_2.npy"
matches_3_save_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/" + features_no + "/matches_3.npy"

# query descs against base model descs
matches_1 = feature_matcher_wrapper_base(db_gt, query_images_names, train_descriptors_base, points3D_xyz, Parameters.ratio_test_val, verbose = True)
np.save(matches_1_save_path, matches_1)
matches_2 = feature_matcher_wrapper_live(points3D_scores_1, db_gt, query_images_names, train_descriptors_live, points3D_xyz, Parameters.ratio_test_val, verbose = True)
np.save(matches_2_save_path, matches_2)
matches_3 = feature_matcher_wrapper_live(points3D_scores_2, db_gt, query_images_names, train_descriptors_live, points3D_xyz, Parameters.ratio_test_val, verbose = True)
np.save(matches_3_save_path, matches_3)

# 2: RANSAC Comparison
# for base model matches
matches_1 = np.load(matches_1_save_path)
print("RANSAC")
poses , data = run_comparison(ransac, matches_1, query_images_names)
np.save(Parameters.matches_1_ransac_1_path_poses, poses)
np.save(Parameters.matches_1_ransac_1_path_data, data)
print("PROSAC only lowe's ratio - (lowes_distance_inverse_ratio)")
poses , data = run_comparison(prosac, matches_1, query_images_names, sort_matches_func = sort_matches, val_idx= Parameters.lowes_distance_inverse_ratio_index)
np.save(Parameters.matches_1_prosac_1_path_poses, poses)
np.save(Parameters.matches_1_prosac_1_path_data, data)

# for live model (depends on how many points3D score you are using here, you are going to have the corresponding matches' RANSAC/PROSAC function calls)
matches_2 = np.load(matches_2_save_path) # goes with points3D_scores_1 - yeah confusing...
print("RANSAC")
poses , data = run_comparison(ransac, matches_2, query_images_names)
np.save(Parameters.matches_2_ransac_1_path_poses, poses)
np.save(Parameters.matches_2_ransac_1_path_data, data)
print("RANSAC + dist")
poses , data = run_comparison(ransac_dist, matches_2, query_images_names, points3D_scores_1)
np.save(Parameters.matches_2_ransac_2_path_poses, poses)
np.save(Parameters.matches_2_ransac_2_path_data, data)
print("PROSAC only lowe's ratio - (lowes_distance_inverse_ratio)")
poses , data = run_comparison(prosac, matches_2, query_images_names, sort_matches_func = sort_matches, val_idx=Parameters.lowes_distance_inverse_ratio_index)
np.save(Parameters.matches_2_prosac_1_path_poses, poses)
np.save(Parameters.matches_2_prosac_1_path_data, data)
print("PROSAC only heatmap value - (points3D_score)")
poses , data = run_comparison(prosac, matches_2, query_images_names, sort_matches_func = sort_matches, val_idx=Parameters.points3D_score_index)
np.save(Parameters.matches_2_prosac_2_path_poses, poses)
np.save(Parameters.matches_2_prosac_2_path_data, data)
print("PROSAC reliability scores ratio (r_m/r_n) value - (reliability_score_ratio)")
poses , data = run_comparison(prosac, matches_2, query_images_names, sort_matches_func = sort_matches, val_idx=Parameters.reliability_score_ratio_index)
np.save(Parameters.matches_2_prosac_3_path_poses, poses)
np.save(Parameters.matches_2_prosac_3_path_data, data)
print("PROSAC lowe's ratio (d_n/d_m) / reliability scores ratio (r_m/r_n) value - (custom_score)")
poses , data = run_comparison(prosac, matches_2, query_images_names, sort_matches_func = sort_matches, val_idx=Parameters.custom_score_index)
np.save(Parameters.matches_2_prosac_4_path_poses, poses)
np.save(Parameters.matches_2_prosac_4_path_data, data)
print("PROSAC highest reliability score - (higher_neighbour_score)")
poses , data = run_comparison(prosac, matches_2, query_images_names, sort_matches_func = sort_matches, val_idx=Parameters.higher_neighbour_score_index)
np.save(Parameters.matches_2_prosac_5_path_poses, poses)
np.save(Parameters.matches_2_prosac_5_path_data, data)

matches_3 = np.load(matches_3_save_path) # goes with points3D_scores_2 - yeah confusing...
print("RANSAC")
poses , data = run_comparison(ransac, matches_3, query_images_names)
np.save(Parameters.matches_3_ransac_1_path_poses, poses)
np.save(Parameters.matches_3_ransac_1_path_data, data)
print("RANSAC + dist")
poses , data = run_comparison(ransac_dist, matches_3, query_images_names, points3D_scores_2)
np.save(Parameters.matches_3_ransac_2_path_poses, poses)
np.save(Parameters.matches_3_ransac_2_path_data, data)
print("PROSAC only lowe's ratio - (lowes_distance_inverse_ratio)")
poses , data = run_comparison(prosac, matches_3, query_images_names, sort_matches_func = sort_matches, val_idx=Parameters.lowes_distance_inverse_ratio_index)
np.save(Parameters.matches_3_prosac_1_path_poses, poses)
np.save(Parameters.matches_3_prosac_1_path_data, data)
print("PROSAC only heatmap value - (points3D_score)")
poses , data = run_comparison(prosac, matches_3, query_images_names, sort_matches_func = sort_matches, val_idx=Parameters.points3D_score_index)
np.save(Parameters.matches_3_prosac_2_path_poses, poses)
np.save(Parameters.matches_3_prosac_2_path_data, data)
print("PROSAC reliability scores ratio (r_m/r_n) value - (reliability_score_ratio)")
poses , data = run_comparison(prosac, matches_3, query_images_names, sort_matches_func = sort_matches, val_idx=Parameters.reliability_score_ratio_index)
np.save(Parameters.matches_3_prosac_3_path_poses, poses)
np.save(Parameters.matches_3_prosac_3_path_data, data)
print("PROSAC lowe's ratio (d_n/d_m) / reliability scores ratio (r_m/r_n) value - (custom_score)")
poses , data = run_comparison(prosac, matches_3, query_images_names, sort_matches_func = sort_matches, val_idx=Parameters.custom_score_index)
np.save(Parameters.matches_3_prosac_4_path_poses, poses)
np.save(Parameters.matches_3_prosac_4_path_data, data)
print("PROSAC highest reliability score - (higher_neighbour_score)")
poses , data = run_comparison(prosac, matches_3, query_images_names, sort_matches_func = sort_matches, val_idx=Parameters.higher_neighbour_score_index)
np.save(Parameters.matches_3_prosac_5_path_poses, poses)
np.save(Parameters.matches_3_prosac_5_path_data, data)

# base
matches_1_ransac_1_data = np.load(Parameters.matches_1_ransac_1_path_data)
matches_1_prosac_1_data = np.load(Parameters.matches_1_prosac_1_path_data)

# live
matches_2_ransac_1_data = np.load(Parameters.matches_2_ransac_1_path_data)
matches_2_ransac_2_data = np.load(Parameters.matches_2_ransac_2_path_data)
matches_2_prosac_1_data = np.load(Parameters.matches_2_prosac_1_path_data)
matches_2_prosac_2_data = np.load(Parameters.matches_2_prosac_2_path_data)
matches_2_prosac_3_data = np.load(Parameters.matches_2_prosac_3_path_data)
matches_2_prosac_4_data = np.load(Parameters.matches_2_prosac_4_path_data)
matches_2_prosac_5_data = np.load(Parameters.matches_2_prosac_5_path_data)

matches_3_ransac_1_data = np.load(Parameters.matches_3_ransac_1_path_data)
matches_3_ransac_2_data = np.load(Parameters.matches_3_ransac_2_path_data)
matches_3_prosac_1_data = np.load(Parameters.matches_3_prosac_1_path_data)
matches_3_prosac_2_data = np.load(Parameters.matches_3_prosac_2_path_data)
matches_3_prosac_3_data = np.load(Parameters.matches_3_prosac_3_path_data)
matches_3_prosac_4_data = np.load(Parameters.matches_3_prosac_4_path_data)
matches_3_prosac_5_data = np.load(Parameters.matches_3_prosac_5_path_data)

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# base
print("base matching filtering results")
print("inliers_no | outliers_no | iterations | elapsed_time mean")
print(matches_1_ransac_1_data.mean(axis = 0))
print(matches_1_prosac_1_data.mean(axis = 0))
print()

# live
print("live 2 matching filtering results")
print("inliers_no | outliers_no | iterations | elapsed_time mean")
print(matches_2_ransac_1_data.mean(axis = 0))
print(matches_2_ransac_2_data.mean(axis = 0))
print(matches_2_prosac_1_data.mean(axis = 0))
print(matches_2_prosac_2_data.mean(axis = 0))
print(matches_2_prosac_3_data.mean(axis = 0))
print(matches_2_prosac_4_data.mean(axis = 0))
print(matches_2_prosac_5_data.mean(axis = 0))
print()

print("live 3 matching filtering results")
print("inliers_no | outliers_no | iterations | elapsed_time mean")
print(matches_3_ransac_1_data.mean(axis = 0))
print(matches_3_ransac_2_data.mean(axis = 0))
print(matches_3_prosac_1_data.mean(axis = 0))
print(matches_3_prosac_2_data.mean(axis = 0))
print(matches_3_prosac_3_data.mean(axis = 0))
print(matches_3_prosac_4_data.mean(axis = 0))
print(matches_3_prosac_5_data.mean(axis = 0))
print()

breakpoint()

# 3: Pose refinement
# print("Refining Poses")
#
# # For this point you will need ground truth 2D x,y coordinates (from db, keypoints, or matches from before)
# # an initial [R|t] from RANSAC and 3D points (matches from before)
# ransac_1_poses = np.load(ransac_1_path_poses)
# ransac_1_poses_refined = refine_poses(query_images_names, ransac_1_poses, matches)
#
ransac_1_poses_refined_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/refined_poses_to_evaluate/ransac_1_poses_refined.npy"
# np.save(ransac_1_poses_refined_path, ransac_1_poses_refined)
#
# breakpoint()

# 4: Evaluate Poses against Ground Truth
print("Evaluating Poses")
# get gt poses
gt_poses = {}
query_images = read_images_binary(Parameters.live_model_query_images_path)
for query_image_name in query_images_names:
    gt_pose = get_query_image_gt_pose(query_image_name, query_images)
    if(gt_pose.size != 0):
        gt_poses[query_image_name] = gt_pose

ransac_1_poses_refined = np.load(ransac_1_poses_refined_path)
t_error, r_error = pose_evaluate(ransac_1_poses_refined.item(), gt_poses)
print("Errors Mean: Trans: %2.1f, Rotations: %2.1f" % (t_error.mean(), r_error.mean()))


