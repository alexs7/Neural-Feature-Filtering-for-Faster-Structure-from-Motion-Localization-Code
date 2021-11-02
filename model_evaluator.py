import os
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

# Need to run "prepare_comparison_data.py" before this file, and that the "random_percentage" matches the one from "prepare_comparison_data.py" results

# This script will read anything that has "Extended" in it's name. This was the best performing model name and I kept it. So when you train another model
# with a different name make sure before you run this script the name is changed to "Extended" - (18/07/2021), TODO: this needs to fixed
# 08/08/2021 changed to "Extended_New_..."
# 01/11/2021 changed to "Extended..."

# use CMU_data dir or Coop_data
# example command (comment and uncomment):
# "CMU" and "slice3" are the data used for training
# python3 model_evaluator.py colmap_data/CMU_data/slice3/ CMU slice3 early_stop_model 5
# "CMU" and "all" are the data used for training
# python3 model_evaluator.py colmap_data/CMU_data/slice3/ all slice3 early_stop_model 5 -> This is the case for the network trained on all the CMU data
# or for Coop
# python3 model_evaluator.py colmap_data/Coop_data/slice1/ Coop slice1 early_stop_model
# TODO: For this code in this file you have to use the container 'ar2056_bath2020ssh' in weatherwax, ssh root@172.17.0.13 (or whatever IP it is)
# This is because the method predict_on_batch() needs the GPUs for speed - make sure they are free too.
# If you test multiple datasets, slice4, slice3, run the in sequence as prediction time will be slower if ran in parallel - RUN the evaluator for all datasets on 1 machine!
# Look at the command below:
# you can run this on any GPU machine as long as it is free:  ar2056_trainNN_GPU_5 etc etc...
# python3 model_evaluator.py colmap_data/CMU_data/slice3/ CMU slice3 early_stop_model 5 && python3 model_evaluator.py colmap_data/CMU_data/slice4/ CMU slice4 early_stop_model 5 && python3 model_evaluator.py colmap_data/CMU_data/slice6/ CMU slice6 early_stop_model 5 && python3 model_evaluator.py colmap_data/CMU_data/slice10/ CMU slice10 early_stop_model 5 && python3 model_evaluator.py colmap_data/CMU_data/slice11/ CMU slice11 early_stop_model 5 && python3 model_evaluator.py colmap_data/Coop_data/slice1/ Coop slice1 early_stop_model 5
# After this script ran: print_eval_NN_results.py to aggregate the results.

base_path = sys.argv[1]
models_dir = "colmap_data/tensorboard_results"
dataset = sys.argv[2]
slice = sys.argv[3]
model = sys.argv[4]
# percentage number 5%, 10%, 20% etc (08/08/2021 - use only 10% for paper)
random_percentage = int(sys.argv[5])
ml_path = os.path.join(base_path, "ML_data")
prepared_data_path = os.path.join(ml_path, "prepared_data")

print("Base path: " + base_path)

class_model_dir =  os.path.join(os.path.join(models_dir, "classification_Extended_"+dataset+"_"+slice), model)
regression_score_per_image_dir = os.path.join(os.path.join(models_dir, "regression_Extended_"+dataset+"_"+slice+"_score_per_image"), model)
regression_all_score_per_image_model_dir = os.path.join(os.path.join(models_dir, "regression_AllExtended_"+dataset+"_"+slice+"_score_per_image"), model)
regression_score_per_session_dir = os.path.join(os.path.join(models_dir, "regression_Extended_"+dataset+"_"+slice+"_score_per_session"), model)
regression_all_score_per_session_model_dir = os.path.join(os.path.join(models_dir, "regression_AllExtended_"+dataset+"_"+slice+"_score_per_session"), model)
regression_score_visibility_model_dir = os.path.join(os.path.join(models_dir, "regression_Extended_"+dataset+"_"+slice+"_score_visibility"), model)
regression_all_visibility_score_visibility_model_dir = os.path.join(os.path.join(models_dir, "regression_AllExtended_"+dataset+"_"+slice+"_score_visibility"), model)
combined_model_score_per_image_dir = os.path.join(os.path.join(models_dir, "combined_Extended_"+dataset+"_"+slice+"_score_per_image"), model)
combined_model_score_per_session_dir = os.path.join(os.path.join(models_dir, "combined_Extended_"+dataset+"_"+slice+"_score_per_session"), model)
combined_model_score_visibility_dir = os.path.join(os.path.join(models_dir, "combined_Extended_"+dataset+"_"+slice+"_score_visibility"), model)

print("Loading Model(s)..")
classification_model = keras.models.load_model(class_model_dir)
regression_model_score_per_image = keras.models.load_model(regression_score_per_image_dir)
regression_model_score_per_session = keras.models.load_model(regression_score_per_session_dir)
regression_model_score_visibility = keras.models.load_model(regression_score_visibility_model_dir)
# trained on all matches
regression_on_all_model_score_per_image = keras.models.load_model(regression_all_score_per_image_model_dir)
regression_on_all_model_score_per_session = keras.models.load_model(regression_all_score_per_session_model_dir)
regression_on_all_model_score_visibility = keras.models.load_model(regression_all_visibility_score_visibility_model_dir)
combined_model_score_per_image = keras.models.load_model(combined_model_score_per_image_dir)
combined_model_score_per_session = keras.models.load_model(combined_model_score_per_session_dir)
combined_model_score_visibility = keras.models.load_model(combined_model_score_visibility_dir)

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

print("Getting matches using classifier only (with top ones selected)..")
matches_cl_top, matching_time_cl_top = feature_matcher_wrapper_model_cl(db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, classifier= classification_model, top_no=random_percentage)
print("Feature Matching time: " + str(matching_time_cl_top))
print()

print("Getting matches using classifier only..")
matches_cl, matching_time_cl = feature_matcher_wrapper_model_cl(db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, classifier= classification_model)
print("Feature Matching time: " + str(matching_time_cl))
print()

print("Getting matches using classifier and regressor (score per images)..")
matches_cl_rg_score_image, matching_time_cl_rg_score_image = feature_matcher_wrapper_model_cl_rg(db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, classification_model, regression_model_score_per_image, random_percentage)
print("Feature Matching time: " + str(matching_time_cl_rg_score_image))
print()

print("Getting matches using classifier and regressor (score per session)..")
matches_cl_rg_score_session, matching_time_cl_rg_score_session = feature_matcher_wrapper_model_cl_rg(db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, classification_model, regression_model_score_per_session, random_percentage)
print("Feature Matching time: " + str(matching_time_cl_rg_score_session))
print()

print("Getting matches using classifier and regressor (score visibility)..")
matches_cl_rg_score_visibility, matching_time_cl_rg_score_visibility = feature_matcher_wrapper_model_cl_rg(db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, classification_model, regression_model_score_visibility, random_percentage)
print("Feature Matching time: " + str(matching_time_cl_rg_score_visibility))
print()

print("Getting matches using regressor (all) only (score per images)..")
matches_rg_score_image, matching_time_rg_score_image = feature_matcher_wrapper_model_rg(db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, regression_on_all_model_score_per_image, random_percentage)
print("Feature Matching time: " + str(matching_time_rg_score_image))
print()

print("Getting matches using regressor (all) only (score per session)..")
matches_rg_score_session, matching_time_rg_score_session = feature_matcher_wrapper_model_rg(db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, regression_on_all_model_score_per_session, random_percentage)
print("Feature Matching time: " + str(matching_time_rg_score_session))
print()

print("Getting matches using regressor (all) only (score visibility)..")
matches_rg_score_visibility, matching_time_rg_score_visibility = feature_matcher_wrapper_model_rg(db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, regression_on_all_model_score_visibility, random_percentage)
print("Feature Matching time: " + str(matching_time_rg_score_visibility))
print()

print("Getting matches using combined NN only (trained on score per image)..")
matches_combined_score_per_image, matching_time_combined_score_per_image = feature_matcher_wrapper_model_cb(db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, combined_model_score_per_image, random_percentage)
print("Feature Matching time: " + str(matching_time_combined_score_per_image))
print()

print("Getting matches using combined NN only (trained on score per session)..")
matches_combined_score_per_session, matching_time_combined_score_per_session = feature_matcher_wrapper_model_cb(db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, combined_model_score_per_session, random_percentage)
print("Feature Matching time: " + str(matching_time_combined_score_per_session))
print()

print("Getting matches using combined NN only (trained on score visibility)..")
matches_combined_score_visibility, matching_time_combined_score_visibility = feature_matcher_wrapper_model_cb(db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, combined_model_score_visibility, random_percentage)
print("Feature Matching time: " + str(matching_time_combined_score_visibility))
print()

print("Benchmarking ML model(s)..")
benchmarks_iters = 3
results = np.empty([0,8])
image_pose_errors_all = []

print("RANSAC.. (classifier only, with top matches selected)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, ransac, matches_cl_top, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_cl_top
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_cl_top))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_cl_top, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("RANSAC.. (classifier only)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, ransac, matches_cl, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_cl
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_cl))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_cl, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("RANSAC.. (classifier and regressor, score per image)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, ransac, matches_cl_rg_score_image, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_cl_rg_score_image
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_cl_rg_score_image))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_cl_rg_score_image, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("RANSAC.. (classifier and regressor, score per session)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, ransac, matches_cl_rg_score_session, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_cl_rg_score_session
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_cl_rg_score_session))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_cl_rg_score_session, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("RANSAC.. (classifier and regressor, score visibility)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, ransac, matches_cl_rg_score_visibility, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_cl_rg_score_visibility
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_cl_rg_score_visibility))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_cl_rg_score_visibility, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("RANSAC.. (regressor only, score per image)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, ransac, matches_rg_score_image, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_rg_score_image
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_rg_score_image))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_rg_score_image, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("RANSAC.. (regressor only, score per session)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, ransac, matches_rg_score_session, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_rg_score_session
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_rg_score_session))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_rg_score_session, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("RANSAC.. (regressor only, score per visibility)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, ransac, matches_rg_score_visibility, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_rg_score_visibility
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_rg_score_visibility))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_rg_score_visibility, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("RANSAC.. (combined NN per image)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, ransac, matches_combined_score_per_image, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_combined_score_per_image
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_combined_score_per_image))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_combined_score_per_image, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("RANSAC.. (combined NN per session)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, ransac, matches_combined_score_per_session, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_combined_score_per_session
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_combined_score_per_session))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_combined_score_per_session, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("RANSAC.. (combined NN on score visibility)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, ransac, matches_combined_score_visibility, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_combined_score_visibility
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_combined_score_visibility))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_combined_score_visibility, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("RANSAC dist.. (classifier and regressor, score per image)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, ransac_dist, matches_cl_rg_score_image, localised_query_images_names, K, query_images_ground_truth_poses, scale, val_idx=-1, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_cl_rg_score_image
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_cl_rg_score_image))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_cl_rg_score_image, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("RANSAC dist.. (classifier and regressor, score per session)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, ransac_dist, matches_cl_rg_score_session, localised_query_images_names, K, query_images_ground_truth_poses, scale, val_idx=-1, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_cl_rg_score_session
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_cl_rg_score_session))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_cl_rg_score_session, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("RANSAC dist.. (classifier and regressor, score visibility)")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, ransac_dist, matches_cl_rg_score_visibility, localised_query_images_names, K, query_images_ground_truth_poses, scale, val_idx=-1, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_cl_rg_score_visibility
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_cl_rg_score_visibility))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_cl_rg_score_visibility, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

# NOTE: for PROSAC the matches are already sorted so just pass 1, no need to sort them again
print("PROSAC - (regressor only, score per image)")
print()
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, prosac, matches_rg_score_image, localised_query_images_names, K, query_images_ground_truth_poses, scale, val_idx=None, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_rg_score_image
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_rg_score_image))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_rg_score_image, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("PROSAC - (regressor only, score per session)")
print()
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, prosac, matches_rg_score_session, localised_query_images_names, K, query_images_ground_truth_poses, scale, val_idx=None, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_rg_score_session
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_rg_score_session))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_rg_score_session, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("PROSAC - (regressor only, score visibility)")
print()
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, prosac, matches_rg_score_visibility, localised_query_images_names, K, query_images_ground_truth_poses, scale, val_idx=None, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_rg_score_visibility
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_rg_score_visibility))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_rg_score_visibility, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("PROSAC - (combined, score per image)")
print()
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, prosac, matches_combined_score_per_image, localised_query_images_names, K, query_images_ground_truth_poses, scale, val_idx=None, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_combined_score_per_image
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_combined_score_per_image))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_combined_score_per_image, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("PROSAC - (combined, score per session)")
print()
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, prosac, matches_combined_score_per_session, localised_query_images_names, K, query_images_ground_truth_poses, scale, val_idx=None, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_combined_score_per_session
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_combined_score_per_session))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_combined_score_per_session, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("PROSAC - (combined, score per visibility)")
print()
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall, image_pose_errors = benchmark_ml(benchmarks_iters, prosac, matches_combined_score_visibility, localised_query_images_names, K, query_images_ground_truth_poses, scale, val_idx=None, verbose=True)
image_pose_errors_all.append(image_pose_errors)
total_time_model = time + matching_time_combined_score_visibility
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Cons. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_model, time, matching_time_combined_score_visibility))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
results = np.r_[results, np.array([inlers_no, outliers, iterations, time, matching_time_combined_score_visibility, total_time_model, trans_errors_overall, rot_errors_overall]).reshape(1,8)]
print()

print("Loading baseline results for comparison..")
random_matches_data = np.load(os.path.join(prepared_data_path, "random_matches_data_"+str(random_percentage)+".npy")).reshape(1,8)
vanillia_matches_data = np.load(os.path.join(prepared_data_path, "vanillia_matches_data_"+str(random_percentage)+".npy")).reshape(1,8)

results = np.r_[results, random_matches_data]
results = np.r_[results, vanillia_matches_data]
# results = np.around(results, 2) #format to 2 decimal places (28/06/2021 - removed rounding)

np.savetxt(os.path.join(ml_path, "results_evaluator_"+str(random_percentage)+".csv"), results, delimiter=",")
np.save(os.path.join(ml_path, "image_pose_errors_all_"+str(random_percentage)+".npy"), np.array(image_pose_errors_all))