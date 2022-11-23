import os
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_query_images_pose_from_images, get_intrinsics_from_camera_bin
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
from tensorflow import keras
from database import COLMAPDatabase
from feature_matching_generator_ML import feature_matcher_wrapper_model_cl, feature_matcher_wrapper_model_cl_rg, feature_matcher_wrapper_model_rg, \
    feature_matcher_wrapper_model_cb
import numpy as np
from ransac_prosac import ransac, ransac_dist, prosac
from benchmark import benchmark
import sys
from parameters import Parameters

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
# TODO: For this code in this file you have to use the container 'ar2056_bath2020ssh (_ssd)' in weatherwax, ssh root@172.17.0.13 (or whatever IP it is), (updated 19/06/2022)
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
parameters = Parameters(base_path)

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

print("Loading Data..")
db_gt_path = os.path.join(base_path, "gt/database.db")
db_gt = COLMAPDatabase.connect(db_gt_path)  # you need this database to get the query images descs as they do NOT exist in the LIVE db, only in GT db!

# the "gt" here means ground truth (also used as query)
query_images_bin_path = os.path.join(base_path, "gt/model/images.bin")
query_images_path = os.path.join(base_path, "gt/query_name.txt")
query_cameras_bin_path = os.path.join(base_path, "gt/model/cameras.bin")
query_images = read_images_binary(query_images_bin_path)
query_images_names = load_images_from_text_file(query_images_path)
# "avg_descs_xyz_ml.npy" is generated by "get_points_3D_mean_desc_single_model_ml.py"
points3D_info = np.load(os.path.join(ml_path, "avg_descs_xyz_ml.npy")).astype(np.float32)
train_descriptors_live = points3D_info[:, 0:128]
localised_query_images_names = get_localised_image_by_names(query_images_names, query_images_bin_path)
query_images_ground_truth_poses = get_query_images_pose_from_images(localised_query_images_names, query_images)
points3D_xyz_live = points3D_info[:,128:132]
K = get_intrinsics_from_camera_bin(query_cameras_bin_path, 3)  # 3 because 1 -base, 2 -live, 3 -query images

# evaluation starts here
print("Feature matching using my models..")
ratio_test_val = 1  # 0.9 as previous publication, 1.0 to test all features (no ratio test)

ml_models_trained = parameters.ml_models_trained #11 in total, NOT the random or baseline
assert(len(ml_models_trained) == 11)

ml_models_trained_idx = 0
print(f"Getting matches using {ml_models_trained[ml_models_trained_idx]}..")
matches_cl_top, images_matching_time, images_percentage_reduction = feature_matcher_wrapper_model_cl(base_path, ml_path, ml_models_trained_idx, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, classifier= classification_model, top_no=random_percentage)
np.save(os.path.join(ml_path, f"images_matching_time_{ml_models_trained_idx}.npy"), images_matching_time)
np.save(os.path.join(ml_path, f"images_percentage_reduction_{ml_models_trained_idx}.npy"), images_percentage_reduction)
ml_models_trained_idx += 1

print(f"Getting matches using {ml_models_trained[ml_models_trained_idx]}..")
matches_cl, images_matching_time, images_percentage_reduction = feature_matcher_wrapper_model_cl(base_path, ml_path, ml_models_trained_idx, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, classifier= classification_model)
np.save(os.path.join(ml_path, f"images_matching_time_{ml_models_trained_idx}.npy"), images_matching_time)
np.save(os.path.join(ml_path, f"images_percentage_reduction_{ml_models_trained_idx}.npy"), images_percentage_reduction)
ml_models_trained_idx += 1

print(f"Getting matches using {ml_models_trained[ml_models_trained_idx]}..")
matches_cl_rg_score_image, images_matching_time, images_percentage_reduction = feature_matcher_wrapper_model_cl_rg(base_path, ml_path, ml_models_trained_idx, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, classification_model, regression_model_score_per_image, random_percentage)
np.save(os.path.join(ml_path, f"images_matching_time_{ml_models_trained_idx}.npy"), images_matching_time)
np.save(os.path.join(ml_path, f"images_percentage_reduction_{ml_models_trained_idx}.npy"), images_percentage_reduction)
ml_models_trained_idx += 1

print(f"Getting matches using {ml_models_trained[ml_models_trained_idx]}..")
matches_cl_rg_score_session, images_matching_time, images_percentage_reduction = feature_matcher_wrapper_model_cl_rg(base_path, ml_path, ml_models_trained_idx, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, classification_model, regression_model_score_per_session, random_percentage)
np.save(os.path.join(ml_path, f"images_matching_time_{ml_models_trained_idx}.npy"), images_matching_time)
np.save(os.path.join(ml_path, f"images_percentage_reduction_{ml_models_trained_idx}.npy"), images_percentage_reduction)
ml_models_trained_idx += 1

print(f"Getting matches using {ml_models_trained[ml_models_trained_idx]}..")
matches_cl_rg_score_visibility, images_matching_time, images_percentage_reduction = feature_matcher_wrapper_model_cl_rg(base_path, ml_path, ml_models_trained_idx, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, classification_model, regression_model_score_visibility, random_percentage)
np.save(os.path.join(ml_path, f"images_matching_time_{ml_models_trained_idx}.npy"), images_matching_time)
np.save(os.path.join(ml_path, f"images_percentage_reduction_{ml_models_trained_idx}.npy"), images_percentage_reduction)
ml_models_trained_idx += 1

print(f"Getting matches using {ml_models_trained[ml_models_trained_idx]}..")
matches_rg_score_image, images_matching_time, images_percentage_reduction = feature_matcher_wrapper_model_rg(base_path, ml_path, ml_models_trained_idx, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, regression_on_all_model_score_per_image, random_percentage)
np.save(os.path.join(ml_path, f"images_matching_time_{ml_models_trained_idx}.npy"), images_matching_time)
np.save(os.path.join(ml_path, f"images_percentage_reduction_{ml_models_trained_idx}.npy"), images_percentage_reduction)
ml_models_trained_idx += 1

print(f"Getting matches using {ml_models_trained[ml_models_trained_idx]}..")
matches_rg_score_session, images_matching_time, images_percentage_reduction = feature_matcher_wrapper_model_rg(base_path, ml_path, ml_models_trained_idx, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, regression_on_all_model_score_per_session, random_percentage)
np.save(os.path.join(ml_path, f"images_matching_time_{ml_models_trained_idx}.npy"), images_matching_time)
np.save(os.path.join(ml_path, f"images_percentage_reduction_{ml_models_trained_idx}.npy"), images_percentage_reduction)
ml_models_trained_idx += 1

print(f"Getting matches using {ml_models_trained[ml_models_trained_idx]}..")
matches_rg_score_visibility, images_matching_time, images_percentage_reduction = feature_matcher_wrapper_model_rg(base_path, ml_path, ml_models_trained_idx, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, regression_on_all_model_score_visibility, random_percentage)
np.save(os.path.join(ml_path, f"images_matching_time_{ml_models_trained_idx}.npy"), images_matching_time)
np.save(os.path.join(ml_path, f"images_percentage_reduction_{ml_models_trained_idx}.npy"), images_percentage_reduction)
ml_models_trained_idx += 1

print(f"Getting matches using {ml_models_trained[ml_models_trained_idx]}..")
matches_combined_score_per_image, images_matching_time, images_percentage_reduction = feature_matcher_wrapper_model_cb(base_path, ml_path, ml_models_trained_idx, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, combined_model_score_per_image, random_percentage)
np.save(os.path.join(ml_path, f"images_matching_time_{ml_models_trained_idx}.npy"), images_matching_time)
np.save(os.path.join(ml_path, f"images_percentage_reduction_{ml_models_trained_idx}.npy"), images_percentage_reduction)
ml_models_trained_idx += 1

print(f"Getting matches using {ml_models_trained[ml_models_trained_idx]}..")
matches_combined_score_per_session, images_matching_time, images_percentage_reduction = feature_matcher_wrapper_model_cb(base_path, ml_path, ml_models_trained_idx, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, combined_model_score_per_session, random_percentage)
np.save(os.path.join(ml_path, f"images_matching_time_{ml_models_trained_idx}.npy"), images_matching_time)
np.save(os.path.join(ml_path, f"images_percentage_reduction_{ml_models_trained_idx}.npy"), images_percentage_reduction)
ml_models_trained_idx += 1

print(f"Getting matches using {ml_models_trained[ml_models_trained_idx]}..")
matches_combined_score_visibility, images_matching_time, images_percentage_reduction = feature_matcher_wrapper_model_cb(base_path, ml_path, ml_models_trained_idx, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, combined_model_score_visibility, random_percentage)
np.save(os.path.join(ml_path, f"images_matching_time_{ml_models_trained_idx}.npy"), images_matching_time)
np.save(os.path.join(ml_path, f"images_percentage_reduction_{ml_models_trained_idx}.npy"), images_percentage_reduction)

assert(ml_models_trained_idx == 10)

# again..
eval_methods = list(parameters.ml_methods.keys())
assert(len(eval_methods) == 20)

print("Benchmarking ML model(s)..") #At this point I combine consensus methods and matches (20 combination in total)

eval_methods_idx = 0
print(f"RANSAC.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(ransac, matches_cl_top, localised_query_images_names, K)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"RANSAC.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(ransac, matches_cl, localised_query_images_names, K)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"RANSAC.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(ransac, matches_cl_rg_score_image, localised_query_images_names, K)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"RANSAC.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(ransac, matches_cl_rg_score_session, localised_query_images_names, K)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"RANSAC.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(ransac, matches_cl_rg_score_visibility, localised_query_images_names, K)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"RANSAC.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(ransac, matches_rg_score_image, localised_query_images_names, K)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"RANSAC.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(ransac, matches_rg_score_session, localised_query_images_names, K)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"RANSAC.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(ransac, matches_rg_score_visibility, localised_query_images_names, K)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"RANSAC.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(ransac, matches_combined_score_per_image, localised_query_images_names, K)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"RANSAC.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(ransac, matches_combined_score_per_session, localised_query_images_names, K)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"RANSAC.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(ransac, matches_combined_score_visibility, localised_query_images_names, K)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"RANSAC dist.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(ransac_dist, matches_cl_rg_score_image, localised_query_images_names, K, val_idx=-1)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"RANSAC dist.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(ransac_dist, matches_cl_rg_score_session, localised_query_images_names, K, val_idx=-1)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"RANSAC dist.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(ransac_dist, matches_cl_rg_score_visibility, localised_query_images_names, K, val_idx=-1)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

# NOTE: for PROSAC the matches are already sorted so just pass 1, no need to sort them again
print(f"PROSAC.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(prosac, matches_rg_score_image, localised_query_images_names, K)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"PROSAC.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(prosac, matches_rg_score_session, localised_query_images_names, K)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"PROSAC.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(prosac, matches_rg_score_visibility, localised_query_images_names, K)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"PROSAC.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(prosac, matches_combined_score_per_image, localised_query_images_names, K)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"PROSAC.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(prosac, matches_combined_score_per_session, localised_query_images_names, K)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)
eval_methods_idx += 1

print(f"PROSAC.. {eval_methods[eval_methods_idx]}")
est_poses_results = benchmark(prosac, matches_combined_score_visibility, localised_query_images_names, K)
np.save(os.path.join(ml_path, f"est_poses_results_{eval_methods_idx}.npy"), est_poses_results)

assert(eval_methods_idx == 19)

print("Done!")
