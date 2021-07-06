import os
import sys

from database import COLMAPDatabase
import numpy as np
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_query_images_pose_from_images, get_intrinsics_from_camera_bin
from point3D_loader import read_points3d_default, get_points3D_xyz
from feature_matching_generator_ML import feature_matcher_wrapper_ml
from ransac_prosac import ransac, ransac_dist, prosac
from get_scale import calc_scale_COLMAP_ARCORE
from benchmark import benchmark, benchmark_ml
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
# percentage number 5%, 10%, 20% etc
random_percentage = int(sys.argv[2])  # Given these features are random the errors later on will be much higher, and benchmarking might fail because there will be < 4 matches sometimes
using_CMU_data = "CMU_data" in base_path
ml_path = os.path.join(base_path, "ML_data")

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
if(using_CMU_data):
    scale = 1
    print("CMU Scale: " + str(scale))
else:
    # for ar_core data
    ar_core_poses_path =  os.path.join(ml_path, "arcore_data/data_all/") #these poses here need to match the frames from the gt images of course
    colmap_poses_path = query_images_bin_path  # just for clarity purposes
    scale = calc_scale_COLMAP_ARCORE(ar_core_poses_path, colmap_poses_path)
    print("ARCore Scale: " + str(scale))

print("Feature matching random and vanillia descs..")
# db_gt, again because we need the descs from the query images
ratio_test_val = 0.9  # 0.9 as previous publication, 1.0 to test all features (no ratio test)

# db_gt is only used to get the SIFT features from the query images, nothing to do with the train_descriptors_live and points3D_xyz_live order. That latter order needs to be corresponding btw.
random_matches, featm_time_random = feature_matcher_wrapper_ml(db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, verbose=True, random_limit=random_percentage)
print("Feature Matching time for random samples (avg per image): " + str(featm_time_random))
# all of them as in first publication (should be around 800 for each image)
vanillia_matches, featm_time_vanillia = feature_matcher_wrapper_ml(db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, verbose=True)
print("Feature Matching time for vanillia samples (avg per image): " + str(featm_time_vanillia))

print()
# get the benchmark data here for random features and the 800 from previous publication - will return the average values for each image
benchmarks_iters = 5 #15 was in first publication

print("Benchmarking Random, iterations: " + str(benchmarks_iters))
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall = benchmark_ml(benchmarks_iters, ransac, random_matches, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
total_time_rand = time + featm_time_random
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Conc. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_rand, time, featm_time_random ))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
random_matches_data = np.array([inlers_no, outliers, iterations, time, featm_time_random, total_time_rand, trans_errors_overall, rot_errors_overall])

print()

print("Benchmarking Vanillia..")
inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall = benchmark_ml(benchmarks_iters, ransac, vanillia_matches, localised_query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
total_time_vanil = time + featm_time_vanillia
print(" Inliers: %2.1f | Outliers: %2.1f | Iterations: %2.1f | Total Time: %2.2f | Conc. Time %2.2f | Feat. M. Time %2.2f " % (inlers_no, outliers, iterations, total_time_vanil, time, featm_time_vanillia ))
print(" Trans Error (m): %2.2f | Rotation Error (Degrees): %2.2f" % (trans_errors_overall, rot_errors_overall))
vanillia_matches_data = np.array([inlers_no, outliers, iterations, time, featm_time_vanillia, total_time_vanil, trans_errors_overall, rot_errors_overall])

prepared_data_path = os.path.join(ml_path, "prepared_data")
os.makedirs(prepared_data_path, exist_ok=True)
np.save(os.path.join(prepared_data_path, "query_images_ground_truth_poses.npy"), query_images_ground_truth_poses)
np.save(os.path.join(prepared_data_path, "localised_query_images_names.npy"), localised_query_images_names)
np.save(os.path.join(prepared_data_path, "points3D_xyz_live.npy"), points3D_xyz_live)
np.save(os.path.join(prepared_data_path, "K.npy"), K)
np.save(os.path.join(prepared_data_path, "scale.npy"), scale)
np.save(os.path.join(prepared_data_path, "random_matches.npy"), random_matches)
np.save(os.path.join(prepared_data_path, "vanillia_matches.npy"), vanillia_matches)
# these below are the files used later in model_evaluator to aggregate all results in one file
np.save(os.path.join(prepared_data_path, "random_matches_data_"+str(random_percentage)+".npy"), random_matches_data)
np.save(os.path.join(prepared_data_path, "vanillia_matches_data_"+str(random_percentage)+".npy"), vanillia_matches_data)
