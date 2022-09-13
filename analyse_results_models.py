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
benchmark_iterations = len(est_poses_results)

# using errors from Benchmarking 6DOF paper (https://www.visuallocalization.net)
trans_errors_all_0 = []
rot_errors_all_0 = []
image_errors_maas = []
image_errors_6dof = []
for benchmark_iteration, data_from_benchmarck_iteration in est_poses_results.items():
    print("analyzing benchmark iteration: " + str(benchmark_iteration))
    # using new metric from (https://www.kaggle.com/code/eduardtrulls/imc2022-training-data#kln-486)
    # this is a good metric for ranking method - remove the z-transform maybe ?
    image_errors_maas += [pose_evaluate_generic_comparison_model_Maa(data_from_benchmarck_iteration, query_images_ground_truth_poses, scale)]
    image_errors_6dof += [pose_evaluate_generic_comparison_model(data_from_benchmarck_iteration, query_images_ground_truth_poses, scale)]

# get the mean maa from benchmark iterations
mean_maa = np.empty([0,benchmark_iterations])
for maa in image_errors_maas:
    mean_maa = np.r_[mean_maa, np.array([np.mean(maa[1]), np.mean(maa[2]), np.mean(maa[3])]).reshape(1,3)]

# mean across the rows (https://stackoverflow.com/questions/40200070/what-does-axis-0-do-in-numpys-sum-function)
# np.array(acc), np.array(acc_q), np.array(acc_t)
mean_maa = np.mean(mean_maa, axis=0)

benchmark_data_from_iteration = image_errors_6dof[0]
image_names = list(benchmark_data_from_iteration.keys())

# get mean translation error (m) and rotation (degrees) for all benchmark iterations
images_errors_6dof_mean = {}
for image_name in image_names:
    errors_6dof_mean = np.empty([0, 2])
    for benchmark_iteration in range(benchmark_iterations):
        errors_6dof_mean = np.r_[errors_6dof_mean, image_errors_6dof[benchmark_iteration][image_name].reshape(1, 2)]
    assert(errors_6dof_mean.shape[0] == 3)
    errors_6dof_mean = np.mean(errors_6dof_mean, axis=0)
    images_errors_6dof_mean[image_name] = errors_6dof_mean

images_benchmark_data_mean = {}
for image_name in image_names:
    # inliers_no, outliers_no, iterations, elapsed_time (time to estimate a pose)
    benchmark_data_mean = np.empty([0, 4])
    for benchmark_iteration in range(benchmark_iterations):
        benchmark_data = est_poses_results[benchmark_iteration][image_name]
        inliers_no = benchmark_data[1]
        outliers_no = benchmark_data[2]
        iterations = benchmark_data[3]
        elapsed_time = benchmark_data[4]
        benchmark_data_mean = np.r_[benchmark_data_mean, np.array([inliers_no, outliers_no, iterations, elapsed_time]).reshape(1, 4)]
    assert(benchmark_data_mean.shape[0] == 3)
    benchmark_data_mean = np.mean(benchmark_data_mean, axis=0)
    images_benchmark_data_mean[image_name] = benchmark_data_mean

