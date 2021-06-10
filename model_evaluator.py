import os

import tensorflow as tf
from tensorflow import keras

from RANSACParameters import RANSACParameters
from database import COLMAPDatabase
from feature_matching_generator_ML import feature_matcher_wrapper_model_cl, feature_matcher_wrapper_model_cl_rg, feature_matcher_wrapper_model_rg
from parameters import Parameters
import numpy as np
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_query_images_pose_from_images, get_intrinsics_from_camera_bin
from point3D_loader import read_points3d_default, get_points3D_xyz
from ransac_prosac import ransac, ransac_dist, prosac
from get_scale import calc_scale_COLMAP_ARCORE
from benchmark import benchmark, benchmark_ml
import sys

# Need to run "prepare_comparison_data.py" before this file
# The models here are the best performing for classification and regression as of 28 May, ManyManyNodesLayersEarlyStopping, and ReversePyramidEarlyStopping
# use CMU_data dir or Coop_data
# example command: "python3 model_evaluator.py colmap_data/CMU_data/slice10/ colmap_data/tensorboard_results/classification_MNMNLCMUsl10/early_stop_model/ colmap_data/tensorboard_results/regression_MNMNLCMUsl10/early_stop_model/ colmap_data/tensorboard_results/combined_MNMNLCMUsl10/early_stop_model/ (or Coop)
# TODO: For this code in this file you have to use the container 'ar2056_bath2020ssh' in weatherwax, ssh root@172.17.0.13 (or whatever IP it is)
# This is because the method predict_on_batch() needs the GPUs for speed - make sure they are free too.
base_path = sys.argv[1]
ml_path = os.path.join(base_path, "ML_data")
class_model_dir = sys.argv[2]
regression_model_dir = sys.argv[3]
combined_model_dir = sys.argv[4]

print("Loading Model(s)..")
classifier_model = keras.models.load_model(class_model_dir)
regression_model = keras.models.load_model(regression_model_dir)
combined_model = keras.models.load_model(combined_model_dir)

db_gt_path = os.path.join(base_path, "gt/database.db")
db_gt = COLMAPDatabase.connect(db_gt_path)  # you need this database to get the query images descs as they do NOT exist in the LIVE db, only in GT db!

# load data generated from "prepare_comparison_data.py"
print("Loading Data..")
points3D_info = np.load(os.path.join(ml_path, "avg_descs_xyz_ml.npy")).astype(np.float32)
train_descriptors_live = points3D_info[:, 0:128]
query_images_ground_truth_poses = np.load(os.path.join(ml_path, "prepared_data/query_images_ground_truth_poses.npy"), allow_pickle=True).item()
localised_query_images_names = np.ndarray.tolist(np.load(os.path.join(ml_path, "prepared_data/localised_query_images_names.npy")))
points3D_xyz_live = np.load(os.path.join(ml_path, "prepared_data/points3D_xyz_live.npy")) # can also pick them up from points3D_info
K = np.load(os.path.join(ml_path, "prepared_data/K.npy"))
scale = np.load(os.path.join(ml_path, "prepared_data/scale.npy"))
# these are not needed now here - paths need updating 07/06/2021
# random_matches = np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/random_matches.npy", allow_pickle=True).item()
# vanillia_matches = np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/vanillia_matches.npy", allow_pickle=True).item()
# random_matches_data = np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/random_matches_data.npy")
# vanillia_matches_data = np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/vanillia_matches_data.npy")

# evaluation starts here
print("Feature matching using model..")
# db_gt, again because we need the descs from the query images
ratio_test_val = 1  # 0.9 as previous publication, 1.0 to test all features (no ratio test)
# top 80 ones - why 80 ?
top_no = 80

print("Getting matches using classifier only..")
matches_cl, matching_time_cl = feature_matcher_wrapper_model_cl(db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, classifier= classifier_model)
print("Feature Matching time: " + str(matching_time_cl))
print()

print("Getting matches using classifier and regressor..")
matches_cl_rg, matching_time_cl_rg = feature_matcher_wrapper_model_cl_rg(db_gt,localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, classifier_model, regression_model, top_no)
print("Feature Matching time: " + str(matching_time_cl_rg))

print("Getting matches using regressor only..")
matches_rg, matching_time_rg = feature_matcher_wrapper_model_rg(db_gt,localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, regression_model, top_no)
print("Feature Matching time: " + str(matching_time_rg))
print()

print("Getting matches using combined NN only..")
matches_combined, matching_time_combined = feature_matcher_wrapper_model_cl(db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, combined_model, top_no)
print("Feature Matching time: " + str(matching_time_combined))
print()

print("Benchmarking ML model(s)..")
benchmarks_iters = 1
val_idx = RANSACParameters.heatmap_val_index #Thas happens to work even though the name makes no sense now

print("RANSAC.. (classifier only)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall = benchmark_ml(benchmarks_iters, ransac, matches_cl, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
total_time_model = time + matching_time_cl
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Conc. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_cl))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
print(" For Excel %2.1f, %2.1f, %2.1f, %2.2f, %2.2f, %2.2f, %2.2f, %2.2f, " % (inlers_no, outliers, iterations, time, matching_time_cl, total_time_model, trans_errors_overall, rot_errors_overall))
# model_matches_data = np.array([inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall]) # do I need this ?
print()

print("RANSAC.. (classifier and regressor)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall = benchmark_ml(benchmarks_iters, ransac, matches_cl_rg, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
total_time_model = time + matching_time_cl_rg
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Conc. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_cl_rg))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
print(" For Excel %2.1f, %2.1f, %2.1f, %2.2f, %2.2f, %2.2f, %2.2f, %2.2f, " % (inlers_no, outliers, iterations, time, matching_time_cl_rg, total_time_model, trans_errors_overall, rot_errors_overall))
# model_matches_data = np.array([inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall]) # do I need this ?
print()

print("RANSAC dist.. (classifier and regressor)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall = benchmark_ml(benchmarks_iters, ransac_dist, matches_cl_rg, localised_query_images_names, K, query_images_ground_truth_poses, scale, val_idx=val_idx, verbose=True)
total_time_model = time + matching_time_cl_rg
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Conc. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_cl_rg))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
print(" For Excel %2.1f, %2.1f, %2.1f, %2.2f, %2.2f, %2.2f, %2.2f, %2.2f, " % (inlers_no, outliers, iterations, time, matching_time_cl_rg, total_time_model, trans_errors_overall, rot_errors_overall))
# model_matches_data = np.array([inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall]) # do I need this ?
print()

print("PROSAC - (regressor, using whatever score the NN was trained on)")
print()
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall = benchmark_ml(benchmarks_iters, prosac, matches_rg, localised_query_images_names, K, query_images_ground_truth_poses, scale, val_idx=val_idx, verbose=True) #val_idx is None here so it passes the already sorted matches to PROSAC in 'ransac_comparison.py'
total_time_model = time + matching_time_rg
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Conc. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_rg))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
print(" For Excel %2.1f, %2.1f, %2.1f, %2.2f, %2.2f, %2.2f, %2.2f, %2.2f, " % (inlers_no, outliers, iterations, time, matching_time_rg, total_time_model, trans_errors_overall, rot_errors_overall))
# model_matches_data = np.array([inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall]) # do I need this ?
print()

print("PROSAC - (combined, using whatever score the NN was trained on)")
print()
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall = benchmark_ml(benchmarks_iters, prosac, matches_combined, localised_query_images_names, K, query_images_ground_truth_poses, scale, val_idx=val_idx, verbose=True) #val_idx is None here so it passes the already sorted matches to PROSAC in 'ransac_comparison.py'
total_time_model = time + matching_time_combined
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Conc. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_combined))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
print(" For Excel %2.1f, %2.1f, %2.1f, %2.2f, %2.2f, %2.2f, %2.2f, %2.2f, " % (inlers_no, outliers, iterations, time, matching_time_combined, total_time_model, trans_errors_overall, rot_errors_overall))
# model_matches_data = np.array([inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall]) # do I need this ?
print()