import tensorflow as tf
from tensorflow import keras
from database import COLMAPDatabase
from parameters import Parameters
import numpy as np
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_query_images_pose_from_images, get_intrinsics_from_camera_bin
from point3D_loader import read_points3d_default, get_points3D_xyz
from feature_matching_generator_ML import feature_matcher_wrapper_model
from ransac_prosac import ransac, ransac_dist, prosac
from get_scale import calc_scale_COLMAP_ARCORE
from benchmark import benchmark, benchmark_ml
import sys

# The models here are the best performing for classification and regression as of 28 May
# example commnad: "python3 model_evaluator.py colmap_data/Coop_data/slice1/ML_data/results/BinaryClassification-ManyManyNodesLayersEarlyStopping-Fri\ May\ 21\ 07\:46\:55\ 2021/early_stop_model/  colmap_data/Coop_data/slice1/ML_data/results/Regression-ManyManyNodesLayersEarlyStopping-Thu\ May\ 27\ 15\:17\:26\ 2021/early_stop_model/"
# TODO: For this code in this file you have to use the container 'ar2056_bath2020ssh' in weatherwax, ssh root@172.17.0.13 (or whatever IP it is)
# This is because the method predict_on_batch() needs the GPUs for speed
class_model_dir = sys.argv[1]
regression_model_dir = sys.argv[2]

print("Loading Model..")
class_model = keras.models.load_model(class_model_dir)
regression_model = keras.models.load_model(regression_model_dir)

db_gt_path = "colmap_data/Coop_data/slice1/ML_data/original_data/gt/database.db"
db_gt = COLMAPDatabase.connect(db_gt_path)  # you need this database to get the query images descs as they do not exist in the live db!

# load data generated from "prepare_comaprison_data.py"
print("Loading Data..")
points3D_info = np.load('colmap_data/Coop_data/slice1/ML_data/avg_descs_xyz_ml.npy').astype(np.float32)
train_descriptors_live = points3D_info[:, 0:128]
query_images_ground_truth_poses = np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/query_images_ground_truth_poses.npy", allow_pickle=True).item()
localised_query_images_names = np.ndarray.tolist(np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/localised_query_images_names.npy"))
points3D_xyz_live = np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/points3D_xyz_live.npy")  # can also pick them up from points3D_info
K = np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/K.npy")
scale = np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/scale.npy")
# these are not needed now here
# random_matches = np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/random_matches.npy", allow_pickle=True).item()
# vanillia_matches = np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/vanillia_matches.npy", allow_pickle=True).item()
# random_matches_data = np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/random_matches_data.npy")
# vanillia_matches_data = np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/vanillia_matches_data.npy")

# evaluation starts here
print("Feature matching using model..")
# db_gt, again because we need the descs from the query images
ratio_test_val = 1  # 0.9 as previous publication, 1.0 to test all features (no ratio test)
# top 80 ones - why 80 ?
first_top = 80  # top or random - here it is top, because I am using the models (run a loop, 400, 80, 40)
second_top = 60 # used for regression

print("Getting matches using classifier only..")
classifier_matches, classifier_feature_matching_time = feature_matcher_wrapper_model(db_gt,localised_query_images_names, train_descriptors_live, points3D_xyz_live,
                                                                                     ratio_test_val, classifier = class_model, regressor = None,
                                                                                     verbose=True, class_top=first_top, regres_top=-1, pick_top_ones=True)
print("Feature Matching time: " + str(classifier_feature_matching_time))

print()

print("Getting matches using classifier and regressor..")
classifier_and_regressor_matches, classifier_and_regressor_feature_matching_time = feature_matcher_wrapper_model(db_gt,localised_query_images_names, train_descriptors_live, points3D_xyz_live,
                                                                                                   ratio_test_val, classifier = class_model, regressor = regression_model, verbose=True,
                                                                                                   class_top=first_top, regres_top=second_top, pick_top_ones=True)
print("Feature Matching time: " + str(classifier_and_regressor_feature_matching_time))

print("Benchmarking ML model(s)..")
benchmarks_iters = 15

print("RANSAC.. (classifier only)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall = benchmark_ml(benchmarks_iters, ransac, classifier_matches, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
total_time_model = time + classifier_feature_matching_time
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Conc. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, classifier_feature_matching_time))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
print(" For Excel %2.1f, %2.1f, %2.1f, %2.2f, %2.2f, %2.2f, %2.2f, %2.2f, " % (inlers_no, outliers, iterations, time, classifier_feature_matching_time, total_time_model, trans_errors_overall, rot_errors_overall))
# model_matches_data = np.array([inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall]) # do I need this ?
print()

print("RANSAC.. (classifier and regressor)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall = benchmark_ml(benchmarks_iters, ransac, classifier_and_regressor_matches, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
total_time_model = time + classifier_and_regressor_feature_matching_time
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Conc. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, classifier_and_regressor_feature_matching_time))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
print(" For Excel %2.1f, %2.1f, %2.1f, %2.2f, %2.2f, %2.2f, %2.2f, %2.2f, " % (inlers_no, outliers, iterations, time, classifier_and_regressor_feature_matching_time, total_time_model, trans_errors_overall, rot_errors_overall))
# model_matches_data = np.array([inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall]) # do I need this ?
print()

print("PROSAC - using score per image regression")
print()
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall = benchmark_ml(benchmarks_iters, prosac, classifier_and_regressor_matches, localised_query_images_names, K, query_images_ground_truth_poses, scale, val_idx=None, verbose=True) #val_idx is None here so it passes the already sorted matches to PROSAC in 'ransac_comparison.py'
total_time_model = time + classifier_and_regressor_feature_matching_time
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Conc. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, classifier_and_regressor_feature_matching_time))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
print(" For Excel %2.1f, %2.1f, %2.1f, %2.2f, %2.2f, %2.2f, %2.2f, %2.2f, " % (inlers_no, outliers, iterations, time, classifier_and_regressor_feature_matching_time, total_time_model, trans_errors_overall, rot_errors_overall))
# model_matches_data = np.array([inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall]) # do I need this ?
print()