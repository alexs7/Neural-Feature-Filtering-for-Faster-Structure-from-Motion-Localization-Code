# This file will generate a csv file with the results of the models.
# then you need to parse that .csv file and generate stats for the thesis
# The .csv file can be parsed with parse_results_for_thesis.py

import csv
import os
import sys
import time

import pycolmap
from tqdm import tqdm

from pose_evaluator import pose_evaluate_generic_comparison_model, pose_evaluate_generic_comparison_model_Maa
from save_2D_points import save_debug_image_simple_ml, save_debug_image_simple_ml_green, save_debug_image_simple_ml_red

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
from tensorflow import keras
import numpy as np
import cv2
from joblib import load
from database import COLMAPDatabase
from parameters import Parameters
from point3D_loader import get_points3D_xyz_id, read_points3d_default
from query_image import get_intrinsics, read_cameras_binary, read_images_binary, load_images_from_text_file, get_localised_images, clear_folder, match, get_keypoints_xy, \
    get_descriptors, get_test_data

# Here all the models are test at once so easier to examine results.
# Report:
# 1 - Avg. Rotation error
# 2 - Avg. Translation error
# 3 - Inliers (not yet)
# 4 - Outliers (not yet)
# 5 - mAA
# 6 - Feature Time
# 7 - Save images and reprojected points
# 8 - Reprojection Error (not yet)
# 9 - 3D points matched to 2D points (not yet)
# 10 - 3D points not matched to 2D points (not yet)
# 11 - Sample Consensus Time
# 12 - Reduction Percentage

ransac_options = pycolmap.RANSACOptions(
    max_error=4.0,  # reprojection error in pixels
    min_inlier_ratio=0.01,
    confidence=0.9999,
    min_num_trials=3000,
    max_num_trials=100000,
)

def run_predictions(model, data, model_type=None):
    mnm_data = data['mnm_data']
    pm_data = data['pm_data']
    nf_data = data['nf_data'] #SIFT + XY + BRG + octave + angle + size + response + dominantOrientation + matched

    # same order as above
    xy_descs = np.c_[nf_data[:,128:130], nf_data[:, 0:128]] #XY + SIFT

    if(model_type == "mnm"):
        mnm_data = mnm_data[:, 0:8].astype(np.float32)
        _, y_pred_mnm = model.predict(mnm_data)  # returns 1 or 0
        matchable_indices = np.where(y_pred_mnm == 1)[0]
        matchable_xy_descs = xy_descs[matchable_indices]
        return matchable_xy_descs
    if(model_type == "nf"):
        # at this point we have all the data for the current image (we don't need the matched value here)
        # we are just predicting, we care about the predictions
        prediction_data = nf_data[:, 0:138] #exclude matched
        y_pred = model.predict(prediction_data, verbose=0)  # returns a value from (0,1)
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        matchable_indices = np.where(y_pred == 1)[0]
        matchable_xy_descs = xy_descs[matchable_indices]  # has to be XY + SIFT
        return matchable_xy_descs
    if (model_type == "nf_small"):
        # at this point we have all the data for the current image (we don't need the matched value here)
        # we are just predicting, we care about the predictions
        prediction_data = nf_data[:, 0:133] #SIFT + XY + BRG
        y_pred = model.predict(prediction_data, verbose=0)  # returns a value from (0,1)
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        matchable_indices = np.where(y_pred == 1)[0]
        matchable_xy_descs = xy_descs[matchable_indices]  # has to be XY + SIFT
        return matchable_xy_descs
    if(model_type == "pm"):
        y_pred = model.predict(pm_data[:,2:130])
        matchable_indices = np.where(y_pred == 1)[0]
        matchable_xy_descs = xy_descs[matchable_indices]  # has to be XY + DESC  # returns 1 or 0
        return matchable_xy_descs

def save_debug_images(parameters, kps_xy, model_img, image_path, matchable_xy_descs_mnm, matchable_xy_descs_nf, matchable_xy_descs_nf_small, matchable_xy_descs_pm):
    source = os.path.join(image_path, model_img.name)

    # TODO: This needs to be refactored
    dest_mnm_overlay = os.path.join(parameters.debug_images_ml_path, "mnm_overlay_" + model_img.name.replace("/", "_"))
    dest_mnm_all = os.path.join(parameters.debug_images_ml_path, "mnm_all_" + model_img.name.replace("/", "_"))
    dest_mnm_matchable_only = os.path.join(parameters.debug_images_ml_path, "mnm_matchable_only_" + model_img.name.replace("/", "_"))

    dest_nf_overlay = os.path.join(parameters.debug_images_ml_path, "nf_overlay_" + model_img.name.replace("/", "_"))
    dest_nf_all = os.path.join(parameters.debug_images_ml_path, "nf_all_" + model_img.name.replace("/", "_"))
    dest_nf_matchable_only = os.path.join(parameters.debug_images_ml_path, "nf_matchable_only_" + model_img.name.replace("/", "_"))

    dest_nf_small_overlay = os.path.join(parameters.debug_images_ml_path, "nf_small_overlay_" + model_img.name.replace("/", "_"))
    dest_nf_small_all = os.path.join(parameters.debug_images_ml_path, "nf_small_all_" + model_img.name.replace("/", "_"))
    dest_nf_small_matchable_only = os.path.join(parameters.debug_images_ml_path, "nf_small_matchable_only_" + model_img.name.replace("/", "_"))

    dest_pm_overlay = os.path.join(parameters.debug_images_ml_path, "pm_overlay_" + model_img.name.replace("/", "_"))
    dest_pm_all = os.path.join(parameters.debug_images_ml_path, "pm_all_" + model_img.name.replace("/", "_"))
    dest_pm_matchable_only = os.path.join(parameters.debug_images_ml_path, "pm_matchable_only_" + model_img.name.replace("/", "_"))

    height, width , _ = cv2.imread(source).shape

    # red on top of green
    save_debug_image_simple_ml(source, kps_xy, matchable_xy_descs_mnm[:, 0:2], f"{dest_mnm_overlay}")
    save_debug_image_simple_ml(source, kps_xy, matchable_xy_descs_nf[:, 0:2], f"{dest_nf_overlay}")
    save_debug_image_simple_ml(source, kps_xy, matchable_xy_descs_nf_small[:, 0:2], f"{dest_nf_small_overlay}")
    save_debug_image_simple_ml(source, kps_xy, matchable_xy_descs_pm[:, 0:2], f"{dest_pm_overlay}")

    # red only (all keypoints)
    save_debug_image_simple_ml_red(source, kps_xy, f"{dest_mnm_all}")
    save_debug_image_simple_ml_red(source, kps_xy, f"{dest_nf_all}")
    save_debug_image_simple_ml_red(source, kps_xy, f"{dest_nf_small_all}")
    save_debug_image_simple_ml_red(source, kps_xy, f"{dest_pm_all}")

    # green only (matchable)
    save_debug_image_simple_ml_green(source, matchable_xy_descs_mnm[:, 0:2], f"{dest_mnm_matchable_only}")
    save_debug_image_simple_ml_green(source, matchable_xy_descs_nf[:, 0:2], f"{dest_nf_matchable_only}")
    save_debug_image_simple_ml_green(source, matchable_xy_descs_nf_small[:, 0:2], f"{dest_nf_small_matchable_only}")
    save_debug_image_simple_ml_green(source, matchable_xy_descs_pm[:, 0:2], f"{dest_pm_matchable_only}")

    return height, width

def get_image_statistics(matchable_xy_descs, train_descriptors_gt, points3D_xyz_ids,
                         Ks, model_img, descs,
                         scale=1, camera_type=None, height=None,
                         width=None):
    # save feature reduction results
    percentage_reduction = 100 - (len(matchable_xy_descs) * 100 / len(descs))

    # Now start the matching
    queryDescriptors = matchable_xy_descs[:, 2:].astype(np.float32)
    keypoints_xy = matchable_xy_descs[:, 0:2]
    ratio_test_val = 1.0

    matches, fm_time = match(queryDescriptors, train_descriptors_gt, keypoints_xy, points3D_xyz_ids, ratio_test_val, k=2)

    # get est pose at this point
    image_points = matches[:, 0:2]
    object_points = matches[:, 2:5]
    K = Ks[model_img.name]

    if(image_points.shape[0] < 4):
        return "Degenerate"

    if(camera_type == 'SIMPLE_PINHOLE'):
        focal_length = K[0,0]
        cx = K[0,2]
        cy = K[1,2]
        camera = pycolmap.Camera(model=camera_type, width=width, height=height, params=[focal_length, cx, cy])
    if (camera_type == 'PINHOLE'):
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        camera = pycolmap.Camera(model=camera_type, width=width, height=height, params=[fx, fy, cx, cy])

    start = time.time()
    est_pose = pycolmap.absolute_pose_estimation(image_points, object_points, camera, max_error_px=4.0)
    end = time.time()
    consensus_time = end - start

    if(est_pose['success'] == False):
        return "Degenerate"

    rotm = pycolmap.qvec_to_rotmat(est_pose['qvec'])
    tvec = est_pose['tvec']
    # est, in camera coordinates
    camera_est_pose = np.c_[rotm, tvec]
    # gt, in camera coordinates
    pose_r = model_img.qvec2rotmat()
    pose_t = model_img.tvec
    camera_gt_pose = np.c_[pose_r, pose_t]
    # calculate the errors
    error_t, error_r = pose_evaluate_generic_comparison_model(camera_est_pose, camera_gt_pose, scale=scale)

    res_data = {}
    res_data["error_rotation"] = error_r
    res_data["error_translation"] = error_t
    res_data["fm_time"] = fm_time
    res_data["consensus_time"] = consensus_time
    res_data["percentage_reduction"] = percentage_reduction
    res_data["est_pose"] = camera_est_pose
    res_data["gt_pose"] = camera_gt_pose
    return res_data

def get_prediction_data(parameters, data):
    # load data
    localised_query_images_mnm = data['localised_query_images_mnm']
    mnm_model = data['mnm_model']
    nf_model = data['nf_model']
    nf_model_small = data['nf_model_small']
    pm_model = data['pm_model']
    image_path = data['gt_image_path']
    points3D_xyz_ids_mnm = data['points3D_xyz_ids_mnm']
    Ks_mnm = data['Ks_mnm']
    scale = data['scale']
    camera_type = data['camera_type']

    # remember this file only works for opencv models
    train_descriptors_gt_mnm = np.load(parameters.avg_descs_gt_path_opencv_mnm).astype(np.float32)

    gt_db_mnm = COLMAPDatabase.connect(parameters.gt_db_path_mnm)
    opencv_data_db = COLMAPDatabase.connect(parameters.ml_database_all_opencv_sift_path)
    predictions_results = {}
    for model_img in tqdm(localised_query_images_mnm.values()):
        img_id = model_img.id

        # The order of test_data
        # test_data[i, 0:128] = sift
        # test_data[i, 128:130] = kps_xy
        # test_data[i, 130] = blue
        # test_data[i, 131] = green
        # test_data[i, 132] = red
        # test_data[i, 133] = octave
        # test_data[i, 134] = angle
        # test_data[i, 135] = size
        # test_data[i, 136] = response
        # test_data[i, 137] = dominantOrientation
        # test_data[i, 138] = matched
        test_data = get_test_data(opencv_data_db, gt_db_mnm, img_id)
        if(test_data is None):
            print("No data for image: ", model_img.name)
            continue

        # all data below is in the same order
        mnm_data = np.c_[test_data[:,128:130], test_data[:,133], test_data[:,134], test_data[:,135], test_data[:,136], test_data[:,131], test_data[:,137], test_data[:,138]]
        pm_data = np.c_[test_data[:,128:130], test_data[:, 0:128]] #NOTE: the xy is not needed here but it doesn't matter
        nf_data = test_data #just for naming purposes

        prediction_data = {}
        prediction_data["mnm_data"] = mnm_data
        prediction_data["nf_data"] = nf_data #also for small
        prediction_data["pm_data"] = pm_data

        matchable_xy_descs_mnm = run_predictions(mnm_model, prediction_data, model_type="mnm")
        matchable_xy_descs_nf = run_predictions(nf_model, prediction_data, model_type="nf")
        matchable_xy_descs_nf_small = run_predictions(nf_model_small, prediction_data, model_type="nf_small")
        matchable_xy_descs_pm = run_predictions(pm_model, prediction_data, model_type="pm")

        kps_xy = test_data[:,128:130]
        descs = test_data[:, 0:128]
        height, width = save_debug_images(parameters, kps_xy, model_img, image_path, matchable_xy_descs_mnm, matchable_xy_descs_nf, matchable_xy_descs_nf_small, matchable_xy_descs_pm)

        # data = {errors_rotation, errors_translation, total_fm_time, total_consencus_time, percentage_reduction_total, est_poses, gt_poses}
        # Degenerate cases are defined below as get_image_statistics returns "Degenerate"
        data_mnm = get_image_statistics(matchable_xy_descs_mnm, train_descriptors_gt_mnm, points3D_xyz_ids_mnm, Ks_mnm, model_img, descs, scale=scale, camera_type=camera_type, height=height, width=width)
        data_nf = get_image_statistics(matchable_xy_descs_nf, train_descriptors_gt_mnm, points3D_xyz_ids_mnm, Ks_mnm, model_img, descs, scale=scale, camera_type=camera_type, height=height, width=width)
        data_nf_small = get_image_statistics(matchable_xy_descs_nf_small, train_descriptors_gt_mnm, points3D_xyz_ids_mnm, Ks_mnm, model_img, descs, scale=scale, camera_type=camera_type, height=height, width=width)
        data_pm = get_image_statistics(matchable_xy_descs_pm, train_descriptors_gt_mnm, points3D_xyz_ids_mnm, Ks_mnm, model_img, descs, scale=scale, camera_type=camera_type, height=height, width=width)

        predictions_results[model_img.name] = {"data_mnm": data_mnm, "data_nf": data_nf, "data_nf_small" : data_nf_small, "data_pm": data_pm}
    return predictions_results

def get_localised_query_images_pose_data_mnm(parameters):
    all_query_images = read_images_binary(parameters.gt_model_images_path_mnm)  # only localised images (but from base,live,gt - we need only gt)
    all_query_images_names = load_images_from_text_file(parameters.query_gt_images_txt_path_mnm)  # only gt images (all)

    localised_qt_images = get_localised_images(all_query_images_names, parameters.gt_model_images_path_mnm)  # only gt images (localised only)
    assert len(localised_qt_images) <= len(all_query_images_names)

    cameras_bin = read_cameras_binary(parameters.gt_model_cameras_path_mnm)
    Ks = get_intrinsics(all_query_images, cameras_bin)

    return localised_qt_images, Ks

def parse_row_data(images_pose_data, thresholds_q, thresholds_t, scale=1, model=None):
    total_fm_time = []  # feature matching time
    total_consensus_time = []  # RANSAC time
    percentage_reduction_total = []
    errors_rotation = []
    errors_translation = []
    est_poses = {}
    gt_poses = {}
    degenerate_images = []
    for img_name, pose_data in images_pose_data.items():
        key = f"data_{model}"
        data = pose_data[key]
        if data == "Degenerate":
            print(f"Degenerate image: {img_name}")
            degenerate_images.append(img_name)
            continue
        total_fm_time.append(data["fm_time"])
        total_consensus_time.append(data["consensus_time"])
        percentage_reduction_total.append(data["percentage_reduction"])
        errors_rotation.append(data["error_rotation"])
        errors_translation.append(data["error_translation"])
        est_poses[img_name] = data["est_pose"]
        gt_poses[img_name] = data["gt_pose"]

    # at this point calculate the mAA
    mAA = pose_evaluate_generic_comparison_model_Maa(est_poses, gt_poses, thresholds_q, thresholds_t, scale=scale)
    results = {}
    results["percentage_reduction_total_mean"] = np.mean(percentage_reduction_total)
    results["total_fm_time_mean"] = np.mean(total_fm_time)
    results["total_consensus_time_mean"] = np.mean(total_consensus_time)
    results["errors_rotation_mean"] = np.mean(errors_rotation)
    results["errors_translation_mean"] = np.mean(errors_translation)
    results["degenerate_images_no"] = len(degenerate_images)
    return mAA, results

def write_row(method_name, results_dict, mAA, writer):
    percentage_reduction_total = results_dict["percentage_reduction_total_mean"]
    total_fm_time = results_dict["total_fm_time_mean"]
    total_consensus_time = results_dict["total_consensus_time_mean"]
    errors_rotation = results_dict["errors_rotation_mean"]
    errors_translation = results_dict["errors_translation_mean"]
    degenerate_images_no = results_dict["degenerate_images_no"]

    writer.writerow(
        [method_name, f'{errors_translation}', f'{errors_rotation}', f'{percentage_reduction_total}', f'{total_fm_time}', f'{total_consensus_time}',
         f'{mAA[0]:}', f"{degenerate_images_no}"])

    print([method_name, f'{errors_translation:.3f}', f'{errors_rotation:.3f}', f'{percentage_reduction_total:.3f}', f'{total_fm_time:.3f}', f'{total_consensus_time:.3f}',
           f'{mAA[0]:.3f}', f"{degenerate_images_no}"])

def write_predictions(base_path, dataset, gt_image_path,thresholds_q, thresholds_t, writer, mnm_model_name, pm_model_name, ARCore=False):
    header = [dataset, 'Avg T. Err', 'Avg R. Err', 'Red. (%)', 'FM T.', 'Cons. T.', 'mAA']
    writer.writerow(header)
    parameters = Parameters(base_path)
    clear_folder(parameters.debug_images_ml_path)

    if(ARCore):
        scale = np.load(parameters.ARCORE_scale_path).reshape(1)[0]
        camera_type = "SIMPLE_PINHOLE" #Retail
    else:
        scale = 1
        camera_type = "PINHOLE" #CMU/Lamar

    print("Loading models..")
    # For MnM (2020)
    mnm_model_path = os.path.join(base_path, parameters.mnm_path, mnm_model_name)
    mnm_model = cv2.ml.RTrees_load(mnm_model_path)
    # For NF (2023)
    nn_model_path = os.path.join(base_path, "ML_data", "classification_model")
    nf_model = keras.models.load_model(nn_model_path, compile=False)
    # This model only is trained on SIFT XY BRG data
    # For NF (small) (2023)
    nn_model_small_path = os.path.join(base_path, "ML_data", "classification_model_small")
    nf_model_small = keras.models.load_model(nn_model_small_path, compile=False)
    # For PM (2014)
    pm_model_path = os.path.join(base_path, parameters.predicting_matchability_comparison_data, pm_model_name)
    pm_model = load(pm_model_path)

    # localised images info only
    localised_query_images_mnm, Ks_mnm = get_localised_query_images_pose_data_mnm(parameters)
    # points3D of model
    points3D_gt = read_points3d_default(parameters.gt_model_points3D_path_mnm)
    points3D_xyz_ids_mnm = get_points3D_xyz_id(points3D_gt)
    data = {}
    data['localised_query_images_mnm'] = localised_query_images_mnm
    data['mnm_model'] = mnm_model
    data['nf_model'] = nf_model
    data['nf_model_small'] = nf_model_small
    data['pm_model'] = pm_model
    data['gt_image_path'] = gt_image_path
    data['points3D_xyz_ids_mnm'] = points3D_xyz_ids_mnm
    data['points3D_gt'] = points3D_gt
    data['Ks_mnm'] = Ks_mnm
    data['thresholds_q'] = thresholds_q
    data['thresholds_t'] = thresholds_t
    data['scale'] = scale
    data['camera_type'] = camera_type
    # return 3 rows, 1 for each model
    print("Getting images pose data...")
    predictions_results_data = get_prediction_data(parameters, data)

    mAA, results_mnm = parse_row_data(predictions_results_data, thresholds_q, thresholds_t, scale=scale, model="mnm")
    write_row("MnM ", results_mnm, mAA, writer)

    mAA, results_nf = parse_row_data(predictions_results_data, thresholds_q, thresholds_t, scale=scale, model="nf")
    write_row("NF ", results_nf, mAA, writer)

    mAA, results_nf_small = parse_row_data(predictions_results_data, thresholds_q, thresholds_t, scale=scale, model="nf_small")
    write_row("NF(s)", results_nf_small, mAA, writer)

    mAA, results_pm = parse_row_data(predictions_results_data, thresholds_q, thresholds_t, scale=scale, model="pm")
    write_row("PM ", results_pm, mAA, writer)

    writer.writerow("")

def pose_result_cmu(writer):
    root_path = "/media/iNicosiaData/engd_data/"
    slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    # for CMU
    thresholds_q = np.linspace(1, 10, 10)
    thresholds_t = np.geomspace(0.2, 5, 10)  # np.geomspace(0.2, 5, 10), same as np.logspace(np.log10(0.2), np.log10(5), num=10, base=10.0)
    for slice_name in slices_names:
        base_path = os.path.join(root_path, "cmu", f"{slice_name}", "exmaps_data")
        gt_image_path = os.path.join(base_path, "gt", "images")
        print("Base path: " + base_path)
        mnm_model_name = "trained_model_pairs_no_4000.xml"
        pm_model_name = "rforest_1200.joblib"
        write_predictions(base_path, slice_name, gt_image_path, thresholds_q, thresholds_t, writer, mnm_model_name, pm_model_name)

def pose_results_lamar(writer):
    root_path = "/media/iNicosiaData/engd_data/"
    # for LAMAR dataset
    thresholds_q = np.linspace(1, 5, 10)
    thresholds_t = np.linspace(0.1, 0.5, 10)
    for dataset in ["LIN", "CAB", "HGE"]:
        base_path = os.path.join(root_path, "lamar", f"{dataset}_colmap_model")
        gt_image_path = os.path.join(root_path, "lamar", dataset, "sessions", "query_val_phone", "raw_data")
        print("Base path: " + base_path)
        mnm_model_name = "trained_model_pairs_no_10000.xml"
        temp_params = Parameters(base_path)
        pm_model_name = f"rforest_{temp_params.predicting_matchability_comparison_data_lamar_no_samples}.joblib"
        write_predictions(base_path, dataset, gt_image_path, thresholds_q, thresholds_t, writer, mnm_model_name, pm_model_name)

def pose_results_retail_shop(writer):
    root_path = "/media/iNicosiaData/engd_data/"
    # for Retail dataset
    thresholds_q = np.linspace(0.5, 2, 10)
    thresholds_t = np.linspace(0.01, 0.05, 10)
    base_path = os.path.join(root_path, "retail_shop", "slice1")
    dataset = "RetailShop"
    gt_image_path = os.path.join(base_path, "gt", "images")
    mnm_model_name = "trained_model_pairs_no_4000.xml"
    pm_model_name = "rforest_1200.joblib"
    write_predictions(base_path, dataset, gt_image_path, thresholds_q, thresholds_t, writer, mnm_model_name, pm_model_name, ARCore=True)

root_path = "/media/iNicosiaData/engd_data/"
file_name = sys.argv[1]
result_file_output_path = os.path.join(root_path, file_name)

with open(result_file_output_path, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    pose_result_cmu(writer)
    pose_results_retail_shop(writer)
    pose_results_lamar(writer) #remember to use the custom version of OpenCV (export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.8/site-packages/) for higher feature matching limits
