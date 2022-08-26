import os
from joblib import load
from feature_matching_generator_ML_comparison_models import feature_matcher_wrapper_generic_comparison_model
from database import COLMAPDatabase
import numpy as np
from ransac_prosac import ransac
from benchmark import benchmark_generic_comparison_model
import sys

# Read the original file model_evaluator.py for notes.
# Similarly to model_evaluator.py, run in sequence NOT parallel
# This file was added to evaluate Predicting Matchability - PM (2014) paper,
# Match or No Match: Keypoint Filtering based on Matching Probability - MoNM (2020) paper, a vanillia RF - vl_rf
# In this file I get the matches then benchmark, and repeat not like in model_evaluator.py where I get all matches then benchmark

base_path = sys.argv[1]
print("Base path: " + base_path)

ml_path = os.path.join(base_path, "ML_data")
prepared_data_path = os.path.join(ml_path, "prepared_data")

comparison_data_path_PM = os.path.join(base_path, "predicting_matchability_comparison_data")
comparison_data_path_MoNM = os.path.join(base_path, "match_or_no_match_comparison_data")
comparison_data_path_vl_rf = os.path.join(base_path, "ml_models_vanillia_comparison_data")

db_gt_path = os.path.join(base_path, "gt/database.db")
db_gt = COLMAPDatabase.connect(db_gt_path)  # you need this database to get the query images descs as they do NOT exist in the LIVE db, only in GT db!

# load data generated from "prepare_comparison_data.py"
print("Loading Data..")
points3D_info = np.load(os.path.join(ml_path, "avg_descs_xyz_ml.npy")).astype(np.float32)
train_descriptors_live = points3D_info[:, 0:128]
localised_query_images_names = np.ndarray.tolist(np.load(os.path.join(ml_path, "prepared_data/localised_query_images_names.npy")))
points3D_xyz_live = np.load(os.path.join(ml_path, "prepared_data/points3D_xyz_live.npy")) # can also pick them up from points3D_info
K = np.load(os.path.join(ml_path, "prepared_data/K.npy"))
scale = np.load(os.path.join(ml_path, "prepared_data/scale.npy"))

# evaluation starts here
print("Feature matching using models..")
# db_gt, again because we need the descs from the query images
ratio_test_val = 1  # 0.9 as previous publication, 1.0 to test all features (no ratio test)

print("Creating dirs for comparison model..")
os.makedirs(comparison_data_path_PM, exist_ok=True)
os.makedirs(comparison_data_path_MoNM, exist_ok=True)
os.makedirs(comparison_data_path_vl_rf, exist_ok=True)

benchmarks_iters = 3
print("benchmarks_iters set to: " + str(benchmarks_iters))

# NOTE: "model" needs to have a predict method and return predictions 0 and 1, not 0.5 or 0.12 or whatever

print("Getting matches using Predicting Matchability (2014) + loading model..")
model_path = os.path.join(comparison_data_path_PM, "rf_model_PM.joblib")
model = load(model_path)
matches, matching_time = feature_matcher_wrapper_generic_comparison_model(base_path, comparison_data_path_PM, model, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val)
print("Feature Matching time (PM): " + str(matching_time))
np.savetxt(os.path.join(comparison_data_path_PM, "matching_time.txt"), [matching_time])
print(" RANSAC..")
est_poses_results = benchmark_generic_comparison_model(benchmarks_iters, ransac, matches, localised_query_images_names, K, scale)
np.save(os.path.join(comparison_data_path_PM, "est_poses_results.npy"), est_poses_results)

# ----------------------------->

print("Getting matches using Match or No Match: Keypoint Filtering based on Matching Probability + loading model..")
model_path = os.path.join(comparison_data_path_MoNM, "rf_match_no_match_sk.joblib")
model = load(model_path)
matches, matching_time = feature_matcher_wrapper_generic_comparison_model(base_path, comparison_data_path_MoNM, model, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, model_type="MatchNoMatch")
print("Feature Matching time (Match or No Match): " + str(matching_time))
np.savetxt(os.path.join(comparison_data_path_MoNM, "matching_time.txt"), [matching_time])
print(" RANSAC..")
est_poses_results = benchmark_generic_comparison_model(benchmarks_iters, ransac, matches, localised_query_images_names, K, scale)
np.save(os.path.join(comparison_data_path_MoNM, "est_poses_results.npy"), est_poses_results)

# ----------------------------->

print("Getting matches using vanillia RF + loading model..")
model_path = os.path.join(comparison_data_path_vl_rf, "rf_model_vanillia_25_25.joblib")
model = load(model_path)
matches, matching_time = feature_matcher_wrapper_generic_comparison_model(base_path, comparison_data_path_vl_rf, model, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val)
print("Feature Matching time (vanillia RF): " + str(matching_time))
np.savetxt(os.path.join(comparison_data_path_vl_rf, "matching_time.txt"), [matching_time])
print(" RANSAC..")
est_poses_results = benchmark_generic_comparison_model(benchmarks_iters, ransac, matches, localised_query_images_names, K, scale)
np.save(os.path.join(comparison_data_path_vl_rf, "est_poses_results.npy"), est_poses_results)

# ----------------------------->

print("Done!")