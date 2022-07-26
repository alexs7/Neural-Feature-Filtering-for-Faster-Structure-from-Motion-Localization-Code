import os

from feature_matching_generator_ML_match_or_no_match import feature_matcher_wrapper_match_or_no_match

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
from tensorflow import keras
from RANSACParameters import RANSACParameters
from database import COLMAPDatabase
from feature_matching_generator_ML import feature_matcher_wrapper_model_cl, feature_matcher_wrapper_model_cl_rg, feature_matcher_wrapper_model_rg, \
    feature_matcher_wrapper_model_cb
from parameters import Parameters
import numpy as np
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_query_images_pose_from_images, get_intrinsics_from_camera_bin
from point3D_loader import read_points3d_default, get_points3D_xyz
from ransac_prosac import ransac, ransac_dist, prosac
from get_scale import calc_scale_COLMAP_ARCORE
from benchmark import benchmark, benchmark_ml
import sys

# Read the original file model_evaluator.py for notes.
# This file was added to evaluate Match or No Match: Keypoint Filtering based on Matching Probability (2020) paper

base_path = sys.argv[1]
models_dir = "colmap_data/tensorboard_results"
dataset = sys.argv[2]
slice = sys.argv[3]
model = sys.argv[4]
# percentage number 5%, 10%, 20% etc (08/08/2021 - use only 10% for paper)
random_percentage = int(sys.argv[5])
ml_path = os.path.join(base_path, "ML_data")
prepared_data_path = os.path.join(ml_path, "prepared_data")
comparison_data_path = os.path.join(base_path, "match_or_no_match_comparison_data")
match_or_no_match_tool_path = "code_to_compare/Match-or-no-match-Keypoint-filtering-based-on-matching-probability/build"

print("Base path: " + base_path)

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

# evaluation starts here
print("Feature matching using models..")
# db_gt, again because we need the descs from the query images
ratio_test_val = 1  # 0.9 as previous publication, 1.0 to test all features (no ratio test)

print("Creating dirs for for Match or No Match: Keypoint Filtering based on Matching Probability (2020) files..")
os.makedirs(comparison_data_path, exist_ok=True)

print("Getting matches using Match or No Match: Keypoint Filtering based on Matching Probability (2020)..")
matches, matching_time = feature_matcher_wrapper_match_or_no_match(base_path, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val)
print("Feature Matching time: " + str(matching_time))
print()

print("Benchmarking Match or No Match: Keypoint Filtering based on Matching Probability (2020) untrained model..")
benchmarks_iters = 3
results = np.empty([0,8])
image_pose_errors_all = []

print("RANSAC.. using Match or No Match: Keypoint Filtering based on Matching Probability (2020)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, ransac, matches, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1, 8)]
print()

import pdb
pdb.set_trace()

print("Loading baseline results for comparison..")
random_matches_data = np.load(os.path.join(prepared_data_path, "random_matches_data_"+str(random_percentage)+".npy")).reshape(1,8)
vanillia_matches_data = np.load(os.path.join(prepared_data_path, "vanillia_matches_data_"+str(random_percentage)+".npy")).reshape(1,8)

results = np.r_[results, random_matches_data]
results = np.r_[results, vanillia_matches_data]
# results = np.around(results, 2) #format to 2 decimal places (28/06/2021 - removed rounding)

np.savetxt(os.path.join(ml_path, "results_evaluator_"+str(random_percentage)+"_match_or_no_match.csv"), results, delimiter=",")
np.save(os.path.join(ml_path, "image_pose_errors_all_"+str(random_percentage)+"_match_or_no_match.npy"), np.array(image_pose_errors_all))