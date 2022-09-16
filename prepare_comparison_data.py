# This file is the equivalent of model_evaluator.py and model_evaluator_comparison_models.py, but for random and baseline only
import os
import sys
from database import COLMAPDatabase
import numpy as np
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_query_images_pose_from_images, get_intrinsics_from_camera_bin
from point3D_loader import read_points3d_default, get_points3D_xyz
from feature_matching_generator_ML import feature_matcher_wrapper_ml
from ransac_prosac import ransac, ransac_dist, prosac
from get_scale import calc_scale_COLMAP_ARCORE
from benchmark import benchmark
from parameters import Parameters

# sample command to run
# python3 prepare_comparison_data.py colmap_data/CMU_data/slice3/ 5 (Note: you will need to run this first, "get_points_3D_mean_desc_single_model_ml.py")

# multiple ones (uncomment, copy and comment):
# python3 prepare_comparison_data.py colmap_data/Coop_data/slice1 5 & disown
# python3 prepare_comparison_data.py colmap_data/CMU_data/slice3 5 & disown
# python3 prepare_comparison_data.py colmap_data/CMU_data/slice4 5 & disown
# python3 prepare_comparison_data.py colmap_data/CMU_data/slice6 5 & disown
# python3 prepare_comparison_data.py colmap_data/CMU_data/slice10 5 & disown
# python3 prepare_comparison_data.py colmap_data/CMU_data/slice11 5 & disown

# or run in sequence (if you run them all at the same time the runtime will be slower as the CPU will struggle)
# python3 prepare_comparison_data.py colmap_data/Coop_data/slice1 5 && python3 prepare_comparison_data.py colmap_data/CMU_data/slice3 5 && python3 prepare_comparison_data.py colmap_data/CMU_data/slice4 5 && python3 prepare_comparison_data.py colmap_data/CMU_data/slice6 5 && python3 prepare_comparison_data.py colmap_data/CMU_data/slice10 5 && python3 prepare_comparison_data.py colmap_data/CMU_data/slice11 5

# The data generated here will be then later used for evaluating ML models in the model_evaluator.py
# Will also save the random matches and the full (800) mathces for all the query images - no need to infer at every evaluation.

base_path = sys.argv[1]
print("Doing.. " + base_path)
# percentage number 5%, 10%, 20% etc
random_percentage = int(sys.argv[2])  # Given these features are random the errors later on will be much higher, and benchmarking might fail because there will be < 4 matches sometimes
using_CMU_data = "CMU_data" in base_path
ml_path = os.path.join(base_path, "ML_data")
prepared_data_path = os.path.join(ml_path, "prepared_data")

random_output = os.path.join(prepared_data_path, "random_output")
if not os.path.exists(random_output):
    os.makedirs(random_output, exist_ok=True)

baseline_output = os.path.join(prepared_data_path, "baseline_output")
if not os.path.exists(baseline_output):
    os.makedirs(baseline_output, exist_ok=True)

print("Setting up...")
print("using_CMU_data: " + str(using_CMU_data))
# the "gt" here means ground truth (also used as query)
# use the "gt" folder form the previous publication folder, but I add more data now such as the arcore poses folder for retail and the GT poses for CMU.
db_gt_path = os.path.join(base_path, "gt/database.db")
query_images_bin_path = os.path.join(base_path, "gt/model/images.bin")
query_images_path = os.path.join(base_path, "gt/query_name.txt")
query_cameras_bin_path = os.path.join(base_path, "gt/model/cameras.bin")

db_gt = COLMAPDatabase.connect(db_gt_path)  # you need this database to get the query images descs as they do not exist in the live db!
query_images = read_images_binary(query_images_bin_path)
query_images_names = load_images_from_text_file(query_images_path)
localised_query_images_names = get_localised_image_by_names(query_images_names, query_images_bin_path)
query_images_ground_truth_poses = get_query_images_pose_from_images(localised_query_images_names, query_images)

# live points
# Note: you will need to run this first, "get_points_3D_mean_desc_single_model_ml.py" - to get the 3D points avg descs, and corresponding xyz coordinates (128 + 3) from the LIVE model.
# also you will need the scale between the colmap poses and the ARCore poses (for example the 2020-06-22 the 392 images are from morning run - or whatever you use) - ONLY if you use the Coop_data
# Matching will happen from the query (or gt) images, on the live model, otherwise if you use the query (gt) model it will be "cheating"
# as the descriptors from the query images that you are trying to match will already be in the query (or gt) model.
# Only use the query model for ground truth pose errors comparisons.
points3D_info = np.load(os.path.join(ml_path, "avg_descs_xyz_ml.npy")).astype(np.float32)
points3D_xyz_live = points3D_info[:,128:131] # in my publication I used to get the points (x,y,z) seperately from the LIVE model, but here get_points_3D_mean_desc_single_model_ml.py already returns them
train_descriptors_live = points3D_info[:, 0:128]

K = get_intrinsics_from_camera_bin(query_cameras_bin_path, 3)  # 3 because 1 -base, 2 -live, 3 -query images

# scale here is not used here, but I kept it for future references.
if(using_CMU_data):
    scale = 1
    print("CMU Scale: " + str(scale))
else:
    # for ar_core data
    ar_core_poses_path =  os.path.join(ml_path, "arcore_data/data_all/") #these poses here need to match the frames from the gt images of course
    colmap_poses_path = query_images_bin_path  # just for clarity purposes
    scale = calc_scale_COLMAP_ARCORE(ar_core_poses_path, colmap_poses_path)
    print("ARCore Scale: " + str(scale))

print("Feature matching random and vanillia (baseline) descs..")

# db_gt is only used to get the SIFT features from the query images, nothing to do with the train_descriptors_live and points3D_xyz_live order. That latter order needs to be corresponding btw.
ratio_test_val = 0.9  # 0.9 as previous publication, 1.0 to test all features (no ratio test)
# ratio_test_val = 1 #because we use only random features here, if we use a percentage and a ratio test then features will be to few to get a pose (TODO: debug this! / discuss this)
random_matches, images_matching_time, images_percentage_reduction = feature_matcher_wrapper_ml(base_path, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, random_output, random_limit=random_percentage)
np.save(os.path.join(random_output, f"images_matching_time.npy"), images_matching_time)
np.save(os.path.join(random_output, f"images_percentage_reduction.npy"), images_percentage_reduction) # should be 'random_percentage' everywhere

# all of them as in first publication (should be around 800 for each image)
vanillia_matches, images_matching_time, images_percentage_reduction = feature_matcher_wrapper_ml(base_path, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, baseline_output)
np.save(os.path.join(baseline_output, f"images_matching_time.npy"), images_matching_time)
np.save(os.path.join(baseline_output, f"images_percentage_reduction.npy"), images_percentage_reduction) # should be '0' everywhere

# get the benchmark data here for random features and the 800 from previous publication - will return the average values for each image
benchmarks_iters = 1 #15 was in first publication (Does it matter here? 1 or 15 should be the same)

print("Benchmarking Random..") #NOTE: The below will break sometimes at assert(len(matches_for_image) >= 4), because of the randonmness
est_poses_results = benchmark(benchmarks_iters, ransac, random_matches, localised_query_images_names, K)
np.save(os.path.join(random_output, f"est_poses_results.npy"), est_poses_results)

print("Benchmarking Vanillia..")
est_poses_results = benchmark(benchmarks_iters, ransac, vanillia_matches, localised_query_images_names, K)
np.save(os.path.join(baseline_output, f"est_poses_results.npy"), est_poses_results)

print('Done!')