# This file will generate visual data (it is a strip down version of learned_models_pose_data.py)
# which is images with the keypoints and the reprojected points.
# It will need to load the models
# NOTE: it will overwrite the parameters.debug_images_ml_path images! from learned_models_pose_data.py.
# You might want to remove saving images in learned_models_pose_data.py
# TODO: in this file you can also generate the reprojection error for each image ??
# It is for the figures in section Qualitative results
# NOTE: MnM stands for the OpenCV SIFT point cloud, this is repeated across the code

import csv
import os
import sys
import time
import random
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
from query_image import get_intrinsics, read_cameras_binary, read_images_binary, load_images_from_text_file, get_localised_images, clear_folder, match, get_test_data

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

def save_debug_images(parameters, kps_xy, model_img, image_path, matchable_xy_descs_mnm, matchable_xy_descs_nf, matchable_xy_descs_pm, matchable_gt, pixel_size):
    source = os.path.join(image_path, model_img.name)

    # TODO: This needs to be refactored
    dest_mnm_overlay = os.path.join(parameters.debug_images_ml_path, "mnm_overlay_" + model_img.name.replace("/", "_"))
    # dest_mnm_all = os.path.join(parameters.debug_images_ml_path, "mnm_all_" + model_img.name.replace("/", "_"))
    # dest_mnm_matchable_only = os.path.join(parameters.debug_images_ml_path, "mnm_matchable_only_" + model_img.name.replace("/", "_"))

    dest_nf_overlay = os.path.join(parameters.debug_images_ml_path, "nf_overlay_" + model_img.name.replace("/", "_"))
    # dest_nf_all = os.path.join(parameters.debug_images_ml_path, "nf_all_" + model_img.name.replace("/", "_"))
    # dest_nf_matchable_only = os.path.join(parameters.debug_images_ml_path, "nf_matchable_only_" + model_img.name.replace("/", "_"))

    dest_pm_overlay = os.path.join(parameters.debug_images_ml_path, "pm_overlay_" + model_img.name.replace("/", "_"))
    # dest_pm_all = os.path.join(parameters.debug_images_ml_path, "pm_all_" + model_img.name.replace("/", "_"))
    # dest_pm_matchable_only = os.path.join(parameters.debug_images_ml_path, "pm_matchable_only_" + model_img.name.replace("/", "_"))

    # gt
    dest_gt_overlay = os.path.join(parameters.debug_images_ml_path, "gt_overlay_" + model_img.name.replace("/", "_"))
    save_debug_image_simple_ml(source, kps_xy, matchable_gt[:, 0:2], f"{dest_gt_overlay}", pixel_size)

    # red on top of green (ovelay)
    save_debug_image_simple_ml(source, kps_xy, matchable_xy_descs_mnm[:, 0:2], f"{dest_mnm_overlay}", pixel_size)
    save_debug_image_simple_ml(source, kps_xy, matchable_xy_descs_nf[:, 0:2], f"{dest_nf_overlay}", pixel_size)
    save_debug_image_simple_ml(source, kps_xy, matchable_xy_descs_pm[:, 0:2], f"{dest_pm_overlay}", pixel_size)

    # red only (all keypoints)
    # save_debug_image_simple_ml_red(source, kps_xy, f"{dest_mnm_all}", pixel_size)
    # save_debug_image_simple_ml_red(source, kps_xy, f"{dest_nf_all}", pixel_size)
    # save_debug_image_simple_ml_red(source, kps_xy, f"{dest_pm_all}", pixel_size)

    # green only (matchable)
    # save_debug_image_simple_ml_green(source, matchable_xy_descs_mnm[:, 0:2], f"{dest_mnm_matchable_only}", pixel_size)
    # save_debug_image_simple_ml_green(source, matchable_xy_descs_nf[:, 0:2], f"{dest_nf_matchable_only}", pixel_size)
    # save_debug_image_simple_ml_green(source, matchable_xy_descs_pm[:, 0:2], f"{dest_pm_matchable_only}", pixel_size)

def save_images_visual_data(parameters, data):
    # load data
    localised_query_images_mnm = data['localised_query_images_mnm']
    mnm_model = data['mnm_model']
    nf_model = data['nf_model']
    nf_model_small = data['nf_model_small'] #NOTE: 04/04/2023: Not used anymore
    pm_model = data['pm_model']
    image_path = data['gt_image_path']
    pixel_size = data['pixel_size']

    gt_db_mnm = COLMAPDatabase.connect(parameters.gt_db_path_mnm)
    opencv_data_db = COLMAPDatabase.connect(parameters.ml_database_all_opencv_sift_path)

    # # or pick 15 random images, using all is too much
    # rand_keys = random.sample(localised_query_images_mnm.keys(), 15)
    # localised_query_images_mnm_random = {k: localised_query_images_mnm[k] for k in rand_keys}

    for model_img in tqdm(localised_query_images_mnm.values()): #_mnm here is not associated to the learning model but the openCV sift point cloud
        img_id = model_img.id

        specific_names = ["ios_2022-06-25_20.22.11_000/images/99912745746.jpg", "session_9/frame_1592760609892.jpg", "session_4/img_02043_c0_1285949849573018us.jpg"]
        if (model_img.name not in specific_names):
            continue

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
        print("Getting image test data from db...")
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
        matchable_xy_descs_pm = run_predictions(pm_model, prediction_data, model_type="pm")

        print("img: ", model_img.name)
        print("mnm matchable no: ", len(matchable_xy_descs_mnm))
        print("nf matchable no: ", len(matchable_xy_descs_nf))
        print("pm matchable no: ", len(matchable_xy_descs_pm))

        kps_xy = test_data[:,128:130]

        # gt data
        all_kps = np.c_[model_img.xys, model_img.point3D_ids]
        kps_xy_gt = all_kps[:, 0:2]  # red
        matchable_gt = all_kps[all_kps[:, 2] != -1][:, 0:2]  # green
        save_debug_images(parameters, kps_xy_gt, model_img, image_path, matchable_xy_descs_mnm, matchable_xy_descs_nf, matchable_xy_descs_pm, matchable_gt, pixel_size)

        # save positive and negative matchable keypoints numbers in txt for each method
        # total / positive, negative = total - positive
        # the files will have a .jpg and .txt extension  they are actually txt files
        print(f"MnM reduction percentage: {len(matchable_xy_descs_mnm) / len(kps_xy) * 100}")
        print(f"NF reduction percentage: {len(matchable_xy_descs_nf) / len(kps_xy) * 100}")
        print(f"PM reduction percentage: {len(matchable_xy_descs_pm) / len(kps_xy) * 100}")
        np.savetxt(os.path.join(parameters.debug_images_ml_path, model_img.name.replace("/", "_").replace(".jpg","_mnm.txt")), [len(prediction_data["mnm_data"]), len(matchable_xy_descs_mnm)], fmt='%f')
        np.savetxt(os.path.join(parameters.debug_images_ml_path, model_img.name.replace("/", "_").replace(".jpg","_nf.txt")), [len(prediction_data["nf_data"]), len(matchable_xy_descs_nf)], fmt='%f')
        np.savetxt(os.path.join(parameters.debug_images_ml_path, model_img.name.replace("/", "_").replace(".jpg","_pm.txt")), [len(prediction_data["pm_data"]), len(matchable_xy_descs_pm)], fmt='%f')

def get_localised_query_images_pose_data_mnm(parameters):
    all_query_images = read_images_binary(parameters.gt_model_images_path_mnm)  # only localised images (but from base,live,gt - we need only gt)
    all_query_images_names = load_images_from_text_file(parameters.query_gt_images_txt_path_mnm)  # only gt images (all)

    localised_qt_images = get_localised_images(all_query_images_names, parameters.gt_model_images_path_mnm)  # only gt images (localised only)
    assert len(localised_qt_images) <= len(all_query_images_names)

    cameras_bin = read_cameras_binary(parameters.gt_model_cameras_path_mnm)
    Ks = get_intrinsics(all_query_images, cameras_bin)

    return localised_qt_images, Ks

# TODO: Clean up - First task after submission to clean up this code, remove the method below 12/04/2023
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

def save_visual_data_wrapper(base_path, gt_image_path, mnm_model_name, pm_model_name, lamar=False):
    parameters = Parameters(base_path)
    clear_folder(parameters.debug_images_ml_path)

    # increase size for teaser figure
    pixel_size = 5 #for retail and CMU
    if(lamar):
        pixel_size = 8

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
    # MnM stands for the OpenCV SIFT point cloud, this is repeated across the code
    localised_query_images_mnm, Ks_mnm = get_localised_query_images_pose_data_mnm(parameters)
    data = {}
    data['localised_query_images_mnm'] = localised_query_images_mnm
    data['mnm_model'] = mnm_model
    data['nf_model'] = nf_model
    data['nf_model_small'] = nf_model_small
    data['pm_model'] = pm_model
    data['gt_image_path'] = gt_image_path
    data['pixel_size'] = pixel_size

    print("Writing down keypoints on images and saving...")
    save_images_visual_data(parameters, data)

def visual_data_result_cmu(root_path):
    slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                    "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    for slice_name in slices_names:
        base_path = os.path.join(root_path, "cmu", f"{slice_name}", "exmaps_data")
        gt_image_path = os.path.join(base_path, "gt", "images")
        print("Base path: " + base_path)
        mnm_model_name = "trained_model_pairs_no_4000.xml"
        pm_model_name = "rforest_1200.joblib"
        save_visual_data_wrapper(base_path, gt_image_path, mnm_model_name, pm_model_name)

def visual_data_results_lamar(root_path):
    for dataset in ["LIN", "CAB", "HGE"]:
        base_path = os.path.join(root_path, "lamar", f"{dataset}_colmap_model")
        gt_image_path = os.path.join(root_path, "lamar", dataset, "sessions", "query_val_phone", "raw_data")
        print("Base path: " + base_path)
        mnm_model_name = "trained_model_pairs_no_10000.xml"
        temp_params = Parameters(base_path)
        pm_model_name = f"rforest_{temp_params.predicting_matchability_comparison_data_lamar_no_samples}.joblib"
        save_visual_data_wrapper(base_path, gt_image_path, mnm_model_name, pm_model_name, lamar=True)

def visual_data_results_retail_shop(root_path):
    base_path = os.path.join(root_path, "retail_shop", "slice1")
    gt_image_path = os.path.join(base_path, "gt", "images")
    mnm_model_name = "trained_model_pairs_no_4000.xml"
    pm_model_name = "rforest_1200.joblib"
    save_visual_data_wrapper(base_path, gt_image_path, mnm_model_name, pm_model_name)

root_path = "/media/iNicosiaData/engd_data/"
visual_data_results_retail_shop(root_path)
visual_data_result_cmu(root_path)
visual_data_results_lamar(root_path)
