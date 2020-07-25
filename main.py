# Arguments
import cv2
from pathlib import Path
from database import COLMAPDatabase
from feature_matching_generator import feature_matcher_wrapper
from parameters import Parameters
from point3D_loader import read_points3d_default, get_points3D_xyz
import numpy as np
from pose_evaluator import pose_evaluate
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_query_images_pose_from_images
from ransac_comparison import run_comparison, sort_matches
from ransac_prosac import ransac, ransac_dist, prosac

features_no = "1k" # colmap_features_no can be "2k", "1k", "0.5k", "0.25k"
exponential_decay_value = 0.5 # exponential_decay can be any of 0.1 to 0.9

print("Doing path: " + Parameters.base_path)
print()

db_gt = COLMAPDatabase.connect(Parameters.gt_db_path) #this database can be used to get the query images descs and ground truth poses for later pose comparison
# Here by "query" I mean the gt images from the gt model - a bit confusing, but think of these images as new new incoming images
# that the user sends with his mobile device. Now the intrinsics will have to be picked up from COLMAP as COLMAP changes the focal point.. (bug..)
# If it didn't change them I could have used just the ones extracted from ARCore in the ARCore case, and the ones provided by CMU in the CMU case.
all_query_images = read_images_binary(Parameters.gt_model_images_path)
all_query_images_names = load_images_from_text_file(Parameters.query_images_path)
localised_query_images_names = get_localised_image_by_names(all_query_images_names, Parameters.gt_model_images_path)

# Note these are the ground truth query images (not session images) that managed to localise against the LIVE model. Might be a low number.
query_images_names = localised_query_images_names
query_images_ground_truth_poses = get_query_images_pose_from_images(query_images_names, all_query_images)

# by "live model" I mean all the frames from future sessions localised in the base model
points3D = read_points3d_default(Parameters.live_model_points3D_path) # live model's 3d points have more images ids than base
points3D_xyz = get_points3D_xyz(points3D)

scale = 1 # default value
if(Path(Parameters.CMU_scale_path).is_file()):
    scale = np.loadtxt(Parameters.CMU_scale_path).reshape(1)[0]

# train_descriptors_base and train_descriptors_live are self explanatory
# train_descriptors must have the same length as the number of points3D
train_descriptors_base = np.load(Parameters.avg_descs_base_path).astype(np.float32)
train_descriptors_live = np.load(Parameters.avg_descs_live_path).astype(np.float32)

# you can check if it is a distribution by calling, .sum() if it is 1, then it is.
# This can be either heatmap_matrix_avg_points_values_0.5.npy or reliability_scores_0.5.npy
# TODO: Do I need to normalise them ?
points3D_heatmap_scores = np.load(Parameters.points3D_scores_1_path)
points3D_reliability_scores = np.load(Parameters.points3D_scores_2_path)
points3D_live_model_scores = [points3D_heatmap_scores, points3D_reliability_scores]

# 1: Feature matching
print("Feature matching...")

# TIP: Remember we are focusing on the model (and its descs) here so the cases to test are:
# query images , train_descriptors from live model : will match base + query images descs to live_model avg descs -> (this can have multiple cases depending on the points3D score used)
# query images , train_descriptors from base model : will match base + query images descs images descs to base avg descs -> (can only be one case...)

query descs against base model descs
matches_base = feature_matcher_wrapper(db_gt, query_images_names, train_descriptors_base, points3D_xyz, Parameters.ratio_test_val, verbose = True)
np.save(Parameters.matches_base_save_path, matches_base)
matches_live = feature_matcher_wrapper(db_gt, query_images_names, train_descriptors_live, points3D_xyz, Parameters.ratio_test_val, True, points3D_live_model_scores)
np.save(Parameters.matches_live_save_path, matches_live)

matches_base = np.load(Parameters.matches_base_save_path, allow_pickle=True).item()
matches_live = np.load(Parameters.matches_live_save_path, allow_pickle=True).item()

# Print options
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

# 2: RANSAC Comparison
results = {}

print()
print("Base Model")
print(" RANSAC")
poses , data = run_comparison(ransac, matches_base, query_images_names)
trans_errors, rot_errors = pose_evaluate(poses, query_images_ground_truth_poses, scale)
results["ransac_base"] = [poses, data, trans_errors, rot_errors]
print(" Inliers: %1.1f | Outliers: %1.1f | Iterations: %1.1f | Time: %2.2f" % (data.mean(axis=0)[0], data.mean(axis=0)[1], data.mean(axis=0)[2], data.mean(axis=0)[3]))
print(" Trans Error (m): %2.2f | Rotation (Degrees): %2.2f" % (np.nanmean(trans_errors), np.nanmean(rot_errors)))
print()
print(" PROSAC only lowe's ratio - (lowes_distance_inverse_ratio)")
poses , data = run_comparison(prosac, matches_base, query_images_names, val_idx= Parameters.lowes_distance_inverse_ratio_index)
trans_errors, rot_errors = pose_evaluate(poses, query_images_ground_truth_poses, scale)
results["prosac_base"] = [poses, data, trans_errors, rot_errors]
print(" Inliers: %1.1f | Outliers: %1.1f | Iterations: %1.1f | Time: %2.2f" % (data.mean(axis=0)[0], data.mean(axis=0)[1], data.mean(axis=0)[2], data.mean(axis=0)[3]))
print(" Trans Error (m): %2.2f | Rotation (Degrees): %2.2f" % (np.nanmean(trans_errors), np.nanmean(rot_errors)))

# -----

print()
print("Live Model")
print(" RANSAC") #No need to run RANSAC multiple times here as it is not using any of the points3D scores
poses , data = run_comparison(ransac, matches_live, query_images_names)
trans_errors, rot_errors = pose_evaluate(poses, query_images_ground_truth_poses, scale)
results["ransac_live"] = [poses, data, trans_errors, rot_errors]
print(" Inliers: %1.1f | Outliers: %1.1f | Iterations: %1.1f | Time: %2.2f" % (data.mean(axis=0)[0], data.mean(axis=0)[1], data.mean(axis=0)[2], data.mean(axis=0)[3]))
print(" Trans Error (m): %2.2f | Rotation (Degrees): %2.2f" % (np.nanmean(trans_errors), np.nanmean(rot_errors)))
print()
print(" RANSAC + dist")
poses, data = run_comparison(ransac_dist, matches_live, query_images_names, val_idx= Parameters.use_ransac_dist)
trans_errors, rot_errors = pose_evaluate(poses, query_images_ground_truth_poses, scale)
results["ransac_dist_live"] = [poses, data, trans_errors, rot_errors]
print(" Inliers: %1.1f | Outliers: %1.1f | Iterations: %1.1f | Time: %2.2f" % (data.mean(axis=0)[0], data.mean(axis=0)[1], data.mean(axis=0)[2], data.mean(axis=0)[3]))
print(" Trans Error (m): %2.2f | Rotation (Degrees): %2.2f" % (np.nanmean(trans_errors), np.nanmean(rot_errors)))

prosac_value_indices = [ Parameters.lowes_distance_inverse_ratio_index,
                         Parameters.heatmap_val_index,
                         Parameters.reliability_score_index,
                         Parameters.reliability_score_ratio_index,
                         Parameters.custom_score_index,
                         Parameters.higher_neighbour_score_index,
                         Parameters.heatmap_val_ratio_index,
                         Parameters.higher_neighbour_val_index ]
print()
print(" PROSAC versions")
np.seterr(divide='ignore', invalid='ignore', over='ignore') # this is because some matches will have a reliability_score of zero. so you might have a division by zero
for prosac_sort_val in prosac_value_indices:
    poses, data = run_comparison(prosac, matches_live, query_images_names, val_idx= prosac_sort_val)
    trans_errors, rot_errors = pose_evaluate(poses, query_images_ground_truth_poses, scale)
    results["procac_live_"+str(prosac_sort_val)] = [poses, data, trans_errors, rot_errors]
    print(Parameters.prosac_value_titles[prosac_sort_val])
    print("  Inliers: %1.1f | Outliers: %1.1f | Iterations: %1.1f | Time: %2.2f" % (data.mean(axis=0)[0], data.mean(axis=0)[1], data.mean(axis=0)[2], data.mean(axis=0)[3]))
    print("  Trans Error (m): %2.2f | Rotation (Degrees): %2.2f" % (np.nanmean(trans_errors), np.nanmean(rot_errors)))

np.save(Parameters.save_results_path, results)

print()
print("Done with path: " + Parameters.base_path)
