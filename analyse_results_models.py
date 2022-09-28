# This file will be used to analyse results from model_evaluator.py, model_evaluator_comparison_models.py
import csv
import os
import sys
import numpy as np
from pose_evaluator import pose_evaluate_generic_comparison_model, pose_evaluate_generic_comparison_model_Maa
from database import COLMAPDatabase
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_query_images_pose_from_images, get_intrinsics_from_camera_bin
from parameters import Parameters

debug_method = ""

def load_est_poses_results(path):
    # [pose, inliers_no, outliers_no, iterations, elapsed_time]
    return np.load(path, allow_pickle=True).item()

def check_for_degenerate_poses(all_benchmark_data):
    # check for degenerate cases first, i.e images that did not have a pose in any or all benchmarks
    deg_names = []
    for _, b_data in all_benchmark_data.items():
        for image_name, _ in b_data.items():  # can be refactored here
            pose = b_data[image_name][0]
            if (pose is None):
                print(f"{image_name} is a degenerate..")
                deg_names += [image_name]
    return np.unique(np.array(deg_names))

def get_maa_accuracy_for_all_images(est_poses_results):
    image_errors_maas = np.empty([0, 3]) #hold the mean for each benchmark iterations
    all_valid_poses_names = np.array([]) #hold the valid images for each benchmark iterations
    all_images_names = list(est_poses_results[0].keys())
    # these are poses that are None because the method was not able to get a pose.
    # either there were less than 4 matches or RANSAC just didn't converge
    degenerate_names = check_for_degenerate_poses(est_poses_results)
    benchmark_iterations = len(est_poses_results)

    # we need to run "pose_evaluate_generic_comparison_model_Maa" across all benchmarks twice
    # first to get common valid names (common) across the benchmark_iterations
    # For example image_23's pose from benchmark iteration 1 and 2 can be in the thresholds
    # but the pose from iteration 3 not. I discard the image completely as it affect the total average maa.
    all_valid_poses_names_from_all_benchmarks = {}
    # fill it with 0
    for q_img_name in list(query_images_ground_truth_poses.keys()):
        all_valid_poses_names_from_all_benchmarks[q_img_name] = 0
    for benchmark_iteration, data_from_benchmarck_iteration in est_poses_results.items():
        _, valid_poses_names = pose_evaluate_generic_comparison_model_Maa(data_from_benchmarck_iteration, query_images_ground_truth_poses, degenerate_names, scale)
        for v_img in valid_poses_names:
            all_valid_poses_names_from_all_benchmarks[v_img] += 1

    common_image_names = []
    for img_name , frequency in all_valid_poses_names_from_all_benchmarks.items():
        if frequency == benchmark_iterations:
            common_image_names.append(img_name)
        else:
            degenerate_names = np.append(degenerate_names, img_name)

    print(degenerate_names)
    if(debug_method == "M. or no M. 2020" or debug_method == "All (~800)"):
        import pdb
        pdb.set_trace()

    # second to get the MAA using the final degenerate_names (which will be skipped).
    for benchmark_iteration, data_from_benchmarck_iteration in est_poses_results.items():
        # using new metric from (https://www.kaggle.com/code/eduardtrulls/imc2022-training-data#kln-486)
        images_mean_maa, valid_poses_names = pose_evaluate_generic_comparison_model_Maa(data_from_benchmarck_iteration, query_images_ground_truth_poses, degenerate_names, scale)
        # # the valid_poses_names should be found across all benchmark_iteration, not only some!
        image_errors_maas = np.r_[image_errors_maas, np.array([np.mean(images_mean_maa[1]), np.mean(images_mean_maa[2]), np.mean(images_mean_maa[3])]).reshape(1, 3)]
        all_valid_poses_names = np.append(all_valid_poses_names, valid_poses_names)

    degenerate_poses_percentage = len(degenerate_names) / len(all_images_names) * 100
    non_degenerate_poses_percentage = 100 - degenerate_poses_percentage

    # mean across the rows (https://stackoverflow.com/questions/40200070/what-does-axis-0-do-in-numpys-sum-function)
    image_errors_maas_mean = np.mean(image_errors_maas, axis=0) # np.array(acc), np.array(acc_q), np.array(acc_t)
    # this will remove duplicate image names that were added from multiple benchmarks
    # for example 'image_1' can have a valid maa from all benchmarks (n) but we don't need it n times duplicated
    unique_valid_poses_names = np.unique(all_valid_poses_names)
    assert len(unique_valid_poses_names) <= len(all_images_names)

    return image_errors_maas_mean, non_degenerate_poses_percentage, degenerate_poses_percentage, unique_valid_poses_names

# using errors from Benchmarking 6DOF paper (https://www.visuallocalization.net)
def get_6dof_accuracy_for_all_images(est_poses_results, query_images_ground_truth_poses, valid_images):
    image_errors_6dof = {}
    benchmark_iterations = len(est_poses_results)
    images_benchmark_data_mean = {}
    for image_name in valid_images:
        gt_pose = query_images_ground_truth_poses[image_name] #only one ground truth
        benchmark_data_all = np.empty([0, 4])
        error_t_all = []
        error_r_all = []
        for benchmark_iteration_idx, data_from_benchmark_iteration in est_poses_results.items():
            q_pose = data_from_benchmark_iteration[image_name][0]  # [0], need the pose only
            error_t, error_r = pose_evaluate_generic_comparison_model(q_pose, gt_pose, scale)
            if(error_t > 500):
                import pdb
                pdb.set_trace()
            error_t_all += [error_t]
            error_r_all += [error_r]
            # the rest of the data here
            # inliers_no, outliers_no, iterations, elapsed_time (time to estimate a pose)
            benchmark_data = data_from_benchmark_iteration[image_name]
            inliers_no = benchmark_data[1]
            outliers_no = benchmark_data[2]
            iterations = benchmark_data[3]
            elapsed_time = benchmark_data[4]
            benchmark_data_all = np.r_[benchmark_data_all, np.array([inliers_no, outliers_no, iterations, elapsed_time]).reshape(1, 4)]
        image_errors_6dof[f"{image_name}"] = [np.array(error_t_all).mean(), np.array(error_r_all).mean()]
        assert (benchmark_data_all.shape[0] == benchmark_iterations)
        benchmark_data_mean = np.mean(benchmark_data_all, axis=0)
        images_benchmark_data_mean[image_name] = benchmark_data_mean

    # {image: trans_err, rot_err}, [inliers_no, outliers_no, iterations, elapsed_time (time to estimate a pose)]
    return image_errors_6dof, images_benchmark_data_mean

def get_row_data(method, errors_maas_mean, image_errors_6dof_mean,
                 images_benchmark_data_mean, times, percentages,
                 valid_poses_names, non_degenerate_poses_percentage, degenerate_poses_percentage):
    inliers = [] # %
    outliers = [] # %
    iterations = []
    cons_time = []
    feature_m_time = []
    t_error = []
    r_error = []
    percent = []
    total_time = []
    maa = errors_maas_mean[0]
    for image_name in valid_poses_names:
        t_error.append(image_errors_6dof_mean[image_name][0])
        r_error.append(image_errors_6dof_mean[image_name][1])
        feature_m_time.append(times[image_name])
        percent.append(percentages[image_name])
        total_samples = images_benchmark_data_mean[image_name][0] + images_benchmark_data_mean[image_name][1]
        # percentages
        inliers.append(images_benchmark_data_mean[image_name][0] * 100 / total_samples)
        outliers.append(images_benchmark_data_mean[image_name][1] * 100 / total_samples)
        iterations.append(images_benchmark_data_mean[image_name][2])
        cons_time.append(images_benchmark_data_mean[image_name][3])
        total_time.append(images_benchmark_data_mean[image_name][3] + times[image_name])

    data_row = [method, np.mean(inliers), np.mean(outliers), np.mean(iterations), np.mean(total_time),
                np.mean(cons_time), np.mean(feature_m_time), np.mean(t_error), np.mean(r_error),
                np.mean(percent), non_degenerate_poses_percentage, degenerate_poses_percentage, maa]

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

header = ['Model Name', 'Inliers (%)', 'Outliers (%)', 'Iterations', 'Total Time (s)', 'Cons. Time (s)', 'Feat. M. Time (s)', 'Trans Error (m)', 'Rotation Error (d)', 'Keypoints Reduction (%)', 'Non-Degenerate Poses(%)', 'Degenerate Poses(%)', 'MAA']
with open(result_file_output_path, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    # My NNs
    for method_idx in range(len(my_methods)):
        print(f"Doing Method {my_methods[method_idx]}")
        # est_poses_results = load_est_poses_results(os.path.join(ml_path, f"est_poses_results_{method_idx}.npy"))
        #
        # errors_maas_mean, non_degenerate_poses_percentage, degenerate_poses_percentage, valid_poses_names = get_maa_accuracy_for_all_images(est_poses_results)
        # image_errors_6dof, images_benchmark_data_mean = get_6dof_accuracy_for_all_images(est_poses_results, query_images_ground_truth_poses, valid_poses_names)
        #
        # feature_matching_file_index = ml_models_matches_file_index[my_methods[method_idx]]
        # times = np.load(os.path.join(ml_path, f"images_matching_time_{feature_matching_file_index}.npy"), allow_pickle=True).item()
        # percentages = np.load(os.path.join(ml_path, f"images_percentage_reduction_{feature_matching_file_index}.npy"), allow_pickle=True).item()
        #
        # csv_row_data = get_row_data(my_methods[method_idx], errors_maas_mean, image_errors_6dof,
        #                             images_benchmark_data_mean, times, percentages,
        #                             valid_poses_names, non_degenerate_poses_percentage, degenerate_poses_percentage)
        # writer.writerow(csv_row_data)

    # The other papers
    for method in comparison_methods:
        print(f"Doing Method {method}")
        debug_method = method
        path = parameters.comparison_methods[method]
        comparison_path = os.path.join(base_path, path)
        est_poses_results = load_est_poses_results(os.path.join(comparison_path, f"est_poses_results.npy"))

        errors_maas_mean, non_degenerate_poses_percentage, degenerate_poses_percentage, valid_poses_names = get_maa_accuracy_for_all_images(est_poses_results)
        image_errors_6dof, images_benchmark_data_mean = get_6dof_accuracy_for_all_images(est_poses_results, query_images_ground_truth_poses, valid_poses_names)

        times = np.load(os.path.join(comparison_path, f"images_matching_time.npy"), allow_pickle=True).item()
        percentages = np.load(os.path.join(comparison_path, f"images_percentage_reduction.npy"), allow_pickle=True).item()

        csv_row_data = get_row_data(method, errors_maas_mean, image_errors_6dof, images_benchmark_data_mean, times, percentages, valid_poses_names,
                                    non_degenerate_poses_percentage, degenerate_poses_percentage)
        writer.writerow(csv_row_data)

    # Random and Baseline
    for method in baseline_methods:
        print(f"Doing Method {method}")
        debug_method = method
        path = os.path.join(base_path, parameters.baseline_methods[method])
        est_poses_results = load_est_poses_results(os.path.join(path, f"est_poses_results.npy"))

        errors_maas_mean, non_degenerate_poses_percentage, degenerate_poses_percentage, valid_poses_names = get_maa_accuracy_for_all_images(est_poses_results)
        image_errors_6dof, images_benchmark_data_mean = get_6dof_accuracy_for_all_images(est_poses_results, query_images_ground_truth_poses, valid_poses_names)

        times = np.load(os.path.join(path, f"images_matching_time.npy"), allow_pickle=True).item()
        percentages = np.load(os.path.join(path, f"images_percentage_reduction.npy"), allow_pickle=True).item()

        csv_row_data = get_row_data(method, errors_maas_mean, image_errors_6dof, images_benchmark_data_mean, times, percentages, valid_poses_names,
                                    non_degenerate_poses_percentage, degenerate_poses_percentage)
        writer.writerow(csv_row_data)

print("Done!")
