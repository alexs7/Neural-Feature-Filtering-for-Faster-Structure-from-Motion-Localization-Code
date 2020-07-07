# Arguments
import cv2

from database import COLMAPDatabase
from feature_matcher import FeatureMatcherTypes, feature_matcher_factory
from feature_matching_generator import feature_matcher_wrapper
from parameters import Parameters
from point3D_loader import read_points3d_default
import numpy as np

from pose_evaluator import pose_evaluate
from query_image import get_images_names_bin, read_images_binary, get_query_image_global_pose_new_model, get_images_names_from_sessions_numbers
from ransac_comparison import run_comparison, sort_matches
from ransac_prosac import ransac, ransac_dist, prosac

features_no = "1k" # colmap_features_no can be "2k", "1k", "0.5k", "0.25k"
exponential_decay_value = 0.5 # exponential_decay can be any of 0.1 to 0.9

db_path = Parameters.db_path
db = COLMAPDatabase.connect(db_path)
#  no_images_per_session[0] is base images
no_images_per_session = Parameters.no_images_per_session

live_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/images.bin"
live_model_all_images = read_images_binary(live_model_images_path)

base_images_names = get_images_names_from_sessions_numbers([no_images_per_session[0]], db, live_model_all_images) # base = just base images
all_images_names = get_images_names_from_sessions_numbers(no_images_per_session, db, live_model_all_images) #all = query + base images (contains also base images - doesn't really matter here - you can see them as query images too)

# switch between these two: avg_descs_base.npy and avg_descs_all.npy (also try weighted)
# train_descriptors must have the same length as the number of points3D
train_descriptors = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/"+features_no+"/avg_descs_all.npy"
train_descriptors = np.load(train_descriptors).astype(np.float32)

# by "live model" I mean all the frames from future sessions localised in the base model (28/03)
live_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/points3D.bin"
points3D = read_points3d_default(live_model_points3D_path)  # base model's 3D points (same length as complete as we do not add points when localising new points, but different image_ids for each point)

# define matcher
matching_algo = FeatureMatcherTypes.BF  # or FeatureMatcherTypes.FLANN
match_ratio_test = 0.9 #Parameters.kFeatureMatchRatioTest #from graphs 0.7 seems to be the most optimal value
norm_type = cv2.NORM_L2
cross_check = False
matcher = feature_matcher_factory(norm_type, cross_check, match_ratio_test, matching_algo)

# you can check if it is a distribution by calling, .sum() if it is 1, then it is.
points3D_scores = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/" + features_no + "/reliability_scores_" + str(exponential_decay_value) + ".npy")

# # Matches save locations
matches_save_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/" + features_no + "/matches.npy"
#
# 1: Feature matching
# TIP: Remember we are focusing on the model (and its descs) here so the cases to test are:
# query images , train_descriptors from live model : will match base + query images descs to live_model avg descs.
# query images , train_descriptors from base model : will match base + query images descs images descs to base avg descs.
matches = feature_matcher_wrapper(points3D_scores, db, all_images_names, train_descriptors, points3D, matcher, verbose = True)

print("Saving matches...")
# # save the 2D-3D matches
np.save(matches_save_path, matches)

# # 2: RANSAC Comparison
# RANSAC Comparison save locations
ransac_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/ransac_images_pose_" + str(exponential_decay_value) + ".npy"
ransac_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/ransac_data_" + str(exponential_decay_value) + ".npy"
ransac_dist_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/ransac_dist_images_pose_" + str(exponential_decay_value) + ".npy"
ransac_dist_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/ransac_dist_data_" + str(exponential_decay_value) + ".npy"
prosac_1_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/prosac_1_images_pose_" + str(exponential_decay_value) + ".npy"
prosac_1_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/prosac_1_data_" + str(exponential_decay_value) + ".npy"
prosac_2_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/prosac_2_images_pose_" + str(exponential_decay_value) + ".npy"
prosac_2_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/prosac_2_data_" + str(exponential_decay_value) + ".npy"
prosac_3_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/prosac_3_kmeans_images_pose_" + str(exponential_decay_value) + ".npy"
prosac_3_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/prosac_3_kmeans_data_" + str(exponential_decay_value) + ".npy"
prosac_4_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/prosac_4_kmeans_images_pose_" + str(exponential_decay_value) + ".npy"
prosac_4_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/prosac_4_kmeans_data_" + str(exponential_decay_value) + ".npy"

# print("RANSAC + dist")
# poses , data = run_comparison(ransac_dist, matches_save_path, all_images_names, points3D_scores)
# np.save(ransac_dist_path_poses, poses)
# np.save(ransac_dist_path_data, data)
# print("RANSAC")
# poses , data = run_comparison(ransac, matches_save_path, all_images_names)
# np.save(ransac_path_poses, poses)
# np.save(ransac_path_data, data)
# PROSAC just lowe's (this might be more suitable for fundamental/homographies)
print("PROSAC only lowe's ratio")
poses , data = run_comparison(prosac, matches_save_path, all_images_names, sort_matches_func = sort_matches, val_idx=6)
np.save(prosac_1_path_poses, poses)
np.save(prosac_1_path_data, data)
print("PROSAC (only heatmap value)")
poses , data = run_comparison(prosac, matches_save_path, all_images_names, sort_matches_func = sort_matches, val_idx=7)
np.save(prosac_2_path_poses, poses)
np.save(prosac_2_path_data, data)
print("PROSAC reliability scores ratio (r_m/r_n) value")
poses , data = run_comparison(prosac, matches_save_path, all_images_names, sort_matches_func = sort_matches, val_idx=8)
np.save(prosac_3_path_poses, poses)
np.save(prosac_3_path_data, data)
print("PROSAC lowe's ratio (d_n/d_m) / reliability scores ratio (r_m/r_n) value")
poses , data = run_comparison(prosac, matches_save_path, all_images_names, sort_matches_func = sort_matches, val_idx=9)
np.save(prosac_4_path_poses, poses)
np.save(prosac_4_path_data, data)

# data format: inliers_no, outliers_no, iterations, elapsed_time
# ransac_dist_data = np.load(ransac_dist_path_data)
# ransac_data = np.load(ransac_path_data)
prosac_1_data = np.load(prosac_1_path_data)
prosac_2_data = np.load(prosac_2_path_data)
prosac_3_data = np.load(prosac_3_path_data)
prosac_4_data = np.load(prosac_4_path_data)

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
print("inliers_no | outliers_no | iterations | elapsed_time mean")
# print(ransac_dist_data.mean(axis = 0))
# print(ransac_data.mean(axis = 0))
print(prosac_1_data.mean(axis = 0))
print(prosac_2_data.mean(axis = 0))
print(prosac_3_data.mean(axis = 0))
print(prosac_4_data.mean(axis = 0))

breakpoint()

# 3: Evaluate Poses against Ground Truth
vanilla_images_poses = np.load(vanillia_data_path_save_poses).item() # it is a dict!
modified_images_poses = np.load(modified_data_path_save_poses).item() # it is a dict!
prosac_1_images_poses = np.load(prosac_1_data_path_save_poses).item() # it is a dict!
prosac_2_images_poses = np.load(prosac_2_data_path_save_poses).item() # it is a dict!

print("Evaluating Poses")
# get gt poses
gt_poses = {}
complete_model_all_images = read_images_binary(complete_model_images_path)
for image in all_images_names:
    gt_poses[image] = get_query_image_global_pose_new_model(image, complete_model_all_images)

t_error, r_error = pose_evaluate(vanilla_images_poses, gt_poses)
print("RANSAC Errors Mean: Trans: %2.1f, Rotations: %2.1f" % (t_error.mean(), r_error.mean()))
t_error, r_error = pose_evaluate(modified_images_poses, gt_poses)
print("RANSAC + sampling Errors Mean: Trans: %2.1f, Rotations: %2.1f" % (t_error.mean(), r_error.mean()))
breakpoint()
t_error, r_error = pose_evaluate(prosac_1_images_poses, gt_poses)
print("PROSAC ( lowe’s dist ratio * heat map vals) Mean: Trans: %2.1f, Rotations: %2.1f" % (t_error.mean(), r_error.mean()))
t_error, r_error = pose_evaluate(prosac_2_images_poses, gt_poses)
print("PROSAC (lowe’s dist ratio) Mean: Trans: %2.1f, Rotations: %2.1f" % (t_error.mean(), r_error.mean()))
