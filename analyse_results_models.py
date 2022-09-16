# This file will be used to analyse results from model_evaluator.py, model_evaluator_comparison_models.py
import csv
import os
import sys
import numpy as np
from pose_evaluator import pose_evaluate_generic_comparison_model, pose_evaluate_generic_comparison_model_Maa
from database import COLMAPDatabase
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_query_images_pose_from_images, get_intrinsics_from_camera_bin
from parameters import Parameters

def load_est_poses_results(path):
    # [pose, inliers_no, outliers_no, iterations, elapsed_time]
    return np.load(path, allow_pickle=True).item()

def get_maa_accuracy_for_all_images(est_poses_results):
    image_errors_maas = [] #hold the mean for each benchmark iterations
    benchmark_iterations = len(est_poses_results)
    for benchmark_iteration, data_from_benchmarck_iteration in est_poses_results.items():
        # using new metric from (https://www.kaggle.com/code/eduardtrulls/imc2022-training-data#kln-486)
        # this is a good metric for ranking method - remove the z-transform maybe ?
        image_errors_maas += [pose_evaluate_generic_comparison_model_Maa(data_from_benchmarck_iteration, query_images_ground_truth_poses, scale)]

    # get the mean maa from benchmark iterations
    mean_maa = np.empty([0, 3])
    for maa in image_errors_maas:
        mean_maa = np.r_[mean_maa, np.array([np.mean(maa[1]), np.mean(maa[2]), np.mean(maa[3])]).reshape(1, 3)]

    # mean across the rows (https://stackoverflow.com/questions/40200070/what-does-axis-0-do-in-numpys-sum-function)
    # np.array(acc), np.array(acc_q), np.array(acc_t)
    mean_maa = np.mean(mean_maa, axis=0)
    return mean_maa

def get_6dof_accuracy_for_all_images(est_poses_results):
    image_errors_6dof = [] #holds all the individual image errors from all benchmark_iterations
    benchmark_iterations = len(est_poses_results)
    for benchmark_iteration, data_from_benchmarck_iteration in est_poses_results.items():
        # using errors from Benchmarking 6DOF paper (https://www.visuallocalization.net)
        image_errors_6dof += [pose_evaluate_generic_comparison_model(data_from_benchmarck_iteration, query_images_ground_truth_poses, scale)]

    # just get the image names here
    benchmark_data_from_iteration = image_errors_6dof[0]
    image_names = list(benchmark_data_from_iteration.keys())

    # get mean translation error (m) and rotation (degrees) for all benchmark iterations
    images_errors_6dof_mean = {}
    for image_name in image_names:
        errors_6dof_mean = np.empty([0, 2])
        for benchmark_iteration in range(benchmark_iterations):
            errors_6dof_mean = np.r_[errors_6dof_mean, image_errors_6dof[benchmark_iteration][image_name].reshape(1, 2)]
        assert (errors_6dof_mean.shape[0] == benchmark_iterations) #the errors for 1 image from 3 benchmark_iterations (TODO: check for None here ?)
        errors_6dof_mean = np.mean(errors_6dof_mean, axis=0)
        images_errors_6dof_mean[image_name] = errors_6dof_mean

    # the rest of the data here (TODO: check for None here ?)
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
        assert (benchmark_data_mean.shape[0] == benchmark_iterations)
        benchmark_data_mean = np.mean(benchmark_data_mean, axis=0)
        images_benchmark_data_mean[image_name] = benchmark_data_mean

    return images_errors_6dof_mean, images_benchmark_data_mean

def get_row_data(method, mean_maa_accuracy_for_method, images_errors_6dof_mean, images_benchmark_data_mean, times, percentages):
    image_names = list(images_errors_6dof_mean.keys())
    inliers = [] # %
    outliers = [] # %
    iterations = []
    cons_time = []
    feature_m_time = []
    t_error = []
    r_error = []
    percent = []
    total_time = []
    maa = mean_maa_accuracy_for_method[0]
    for image_name in image_names:
        t_error.append(images_errors_6dof_mean[image_name][0])
        r_error.append(images_errors_6dof_mean[image_name][1])
        feature_m_time.append(times[image_name])
        percent.append(percentages[image_name])
        total_samples = images_benchmark_data_mean[image_name][0] + images_benchmark_data_mean[image_name][1]
        inliers.append(images_benchmark_data_mean[image_name][0] * 100 / total_samples)
        outliers.append(images_benchmark_data_mean[image_name][1] * 100 / total_samples)
        iterations.append(images_benchmark_data_mean[image_name][2])
        cons_time.append(images_benchmark_data_mean[image_name][3])
        total_time.append(images_benchmark_data_mean[image_name][3] + times[image_name])

    data_row = [method, np.mean(inliers), np.mean(outliers), np.mean(iterations), np.mean(total_time),
                np.mean(cons_time), np.mean(feature_m_time), np.mean(t_error), np.mean(r_error), np.mean(percent), maa]

    return data_row

base_path = sys.argv[1]
print("Base path: " + base_path)
ml_path = os.path.join(base_path, "ML_data")
result_file_output_path = os.path.join(base_path, "results_2022.csv")

print("Loading Data..")
scale = np.load(os.path.join(ml_path, "prepared_data/scale.npy"))
db_gt_path = os.path.join(base_path, "gt/database.db")
db_gt = COLMAPDatabase.connect(db_gt_path)  # you need this database to get the query images descs as they do NOT exist in the LIVE db, only in GT db!

parameters = Parameters(base_path)
# the "gt" here means ground truth (also used as query)
query_images_bin_path = os.path.join(base_path, "gt/model/images.bin")
query_images_path = os.path.join(base_path, "gt/query_name.txt")
query_cameras_bin_path = os.path.join(base_path, "gt/model/cameras.bin")
query_images = read_images_binary(query_images_bin_path)
query_images_names = load_images_from_text_file(query_images_path)
localised_query_images_names = get_localised_image_by_names(query_images_names, query_images_bin_path)
query_images_ground_truth_poses = get_query_images_pose_from_images(localised_query_images_names, query_images)
K = get_intrinsics_from_camera_bin(query_cameras_bin_path, 3)  # 3 because 1 -base, 2 -live, 3 -query images

# Do my ml_methods first
my_methods = list(parameters.ml_methods.keys())
ml_models_matches_file_index = parameters.ml_methods_matches_map

comparison_methods = list(parameters.comparison_methods.keys())
baseline_methods = list(parameters.baseline_methods.keys())

header = ['Model Name', 'Inliers (%)', 'Outliers (%)', 'Iterations', 'Total Time (s)', 'Cons. Time (s)', 'Feat. M. Time (s)', 'Trans Error (m)', 'Rotation Error (d)', 'Reduction (%)', 'MAA']
with open(result_file_output_path, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    # My NNs
    for method_idx in range(len(my_methods)):
        est_poses_results = load_est_poses_results(os.path.join(ml_path, f"est_poses_results_{method_idx}.npy"))

        mean_maa_accuracy_for_method = get_maa_accuracy_for_all_images(est_poses_results) # np.array(acc), np.array(acc_q), np.array(acc_t)
        images_errors_6dof_mean, images_benchmark_data_mean = get_6dof_accuracy_for_all_images(est_poses_results)

        feature_matching_file_index = ml_models_matches_file_index[my_methods[method_idx]]
        times = np.load(os.path.join(ml_path, f"images_matching_time_{feature_matching_file_index}.npy"), allow_pickle=True).item()
        percentages = np.load(os.path.join(ml_path, f"images_percentage_reduction_{feature_matching_file_index}.npy"), allow_pickle=True).item()

        csv_row_data = get_row_data(my_methods[method_idx], mean_maa_accuracy_for_method, images_errors_6dof_mean, images_benchmark_data_mean, times, percentages)
        writer.writerow(csv_row_data)

    # The other papers
    for method in comparison_methods:
        path = parameters.comparison_methods[method]
        comparison_path = os.path.join(base_path, path)
        est_poses_results = load_est_poses_results(os.path.join(comparison_path, f"est_poses_results.npy"))

        mean_maa_accuracy_for_method = get_maa_accuracy_for_all_images(est_poses_results)  # np.array(acc), np.array(acc_q), np.array(acc_t)
        images_errors_6dof_mean, images_benchmark_data_mean = get_6dof_accuracy_for_all_images(est_poses_results)

        times = np.load(os.path.join(comparison_path, f"images_matching_time.npy"), allow_pickle=True).item()
        percentages = np.load(os.path.join(comparison_path, f"images_percentage_reduction.npy"), allow_pickle=True).item()

        csv_row_data = get_row_data(method, mean_maa_accuracy_for_method, images_errors_6dof_mean, images_benchmark_data_mean, times, percentages)
        writer.writerow(csv_row_data)

    # Random and Baseline
    for method in baseline_methods:
        path = os.path.join(base_path, parameters.baseline_methods[method])
        est_poses_results = load_est_poses_results(os.path.join(path, f"est_poses_results.npy"))

        mean_maa_accuracy_for_method = get_maa_accuracy_for_all_images(est_poses_results)  # np.array(acc), np.array(acc_q), np.array(acc_t)
        images_errors_6dof_mean, images_benchmark_data_mean = get_6dof_accuracy_for_all_images(est_poses_results)

        times = np.load(os.path.join(path, f"images_matching_time.npy"), allow_pickle=True).item()
        percentages = np.load(os.path.join(path, f"images_percentage_reduction.npy"), allow_pickle=True).item()

        csv_row_data = get_row_data(method, mean_maa_accuracy_for_method, images_errors_6dof_mean, images_benchmark_data_mean, times, percentages)
        writer.writerow(csv_row_data)

print("Done!")
