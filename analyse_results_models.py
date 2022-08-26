# This file will be used to analyse results from model_evaluator.py, model_evaluator_comparison_models.py
import os
import sys
import numpy as np
from pose_evaluator import pose_evaluate_generic_comparison_model, pose_evaluate_generic_comparison_model_Maa

def get_matching_times(path):
    return np.loadtxt(os.path.join(path, "matching_time.txt"))

def good_matches_total_all_images(path):
    return np.loadtxt(os.path.join(path, "good_matches_total_all_images.txt"))

def good_matches_avg_per_image(path):
    return np.loadtxt(os.path.join(path, "good_matches_avg_images.txt"))

def percentage_reduction_avg(path):
    return np.loadtxt(os.path.join(path, "percentage_reduction_avg.txt"))

def load_est_poses_results(path):
    # [pose, inliers_no, outliers_no, iterations, elapsed_time]
    return np.load(os.path.join(path, "est_poses_results.npy"), allow_pickle=True).item()

base_path = sys.argv[1]
comparison_data_path = sys.argv[2]
print("Base path: " + base_path)
ml_path = os.path.join(base_path, "ML_data")
scale = np.load(os.path.join(ml_path, "prepared_data/scale.npy"))
query_images_ground_truth_poses = np.load(os.path.join(ml_path, "prepared_data/query_images_ground_truth_poses.npy"), allow_pickle=True).item()

matching_times_pm = get_matching_times(comparison_data_path)
good_matches_total_all_images_pm = good_matches_total_all_images(comparison_data_path)
good_matches_avg_per_image_pm = good_matches_avg_per_image(comparison_data_path)
percentage_reduction_avg_pm = percentage_reduction_avg(comparison_data_path)
est_poses_results = load_est_poses_results(comparison_data_path)

# using errors from Benchmarking 6DOF paper (https://www.visuallocalization.net)
trans_errors_all_0 = []
rot_errors_all_0 = []
image_errors_maas = []
image_errors_6dof = []
for  benchmark_iteration, data_from_benchmarck_iteration in est_poses_results.items():
    print("analyzing benchmark iteration: " + str(benchmark_iteration))
    # using new metric from (https://www.kaggle.com/code/eduardtrulls/imc2022-training-data#kln-486)
    # this is a good metric for ranking method - remove the z-transform maybe ?
    image_errors_maas += [pose_evaluate_generic_comparison_model_Maa(data_from_benchmarck_iteration, query_images_ground_truth_poses, scale)]
    image_errors_6dof += [pose_evaluate_generic_comparison_model(data_from_benchmarck_iteration, query_images_ground_truth_poses, scale)]

# get the mean maa from benchmark iterations
mean_maa = np.empty([0,3])
for maa in image_errors_maas:
    mean_maa = np.r_[mean_maa, np.array([np.mean(maa[1]), np.mean(maa[2]), np.mean(maa[3])]).reshape(1,3)]

# mean across the rows (https://stackoverflow.com/questions/40200070/what-does-axis-0-do-in-numpys-sum-function)
mean_maa = np.mean(mean_maa, axis=0)

# get the mean 6dof poses errors and other benchmarks iteration data from model_evaluator_comparison_models.py
# TODO: save the data in an array here so you can mean later on colums
for image_errors_6dof_benchmark_iteration in range(len(image_errors_6dof)):
    benchmark_data_from_iteration = image_errors_6dof[image_errors_6dof_benchmark_iteration]
    for image_name, errors in benchmark_data_from_iteration.items():
        # benchmark_data is: pose, inliers_no, outliers_no, iterations, elapsed_time, from model_evaluator_comparison_models.py
        benchmark_data = est_poses_results[image_errors_6dof_benchmark_iteration][image_name]
        # translation (m), rotation (degrees)
        errors_6dof = benchmark_data_from_iteration[image_name]

import pdb
pdb.set_trace()
mean_maa = np.mean(mean_maa, axis=1)


