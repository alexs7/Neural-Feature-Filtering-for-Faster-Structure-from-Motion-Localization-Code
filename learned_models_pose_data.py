# Look at learned_models_benchmarks.py for docs
# You can run this file after learned_models_benchmarks.py
#  An note here is that I use the gt model to localise the gt images not the live model.

import csv
import os
import sys
import time

import pycolmap
from tqdm import tqdm

from pose_evaluator import pose_evaluate_generic_comparison_model, pose_evaluate_generic_comparison_model_Maa
from save_2D_points import save_debug_image_simple_ml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
from tensorflow import keras
import numpy as np
import cv2
from joblib import load
from database import COLMAPDatabase
from parameters import Parameters
from point3D_loader import get_points3D_xyz_id, read_points3d_default
from query_image import get_intrinsics, read_cameras_binary, read_images_binary, load_images_from_text_file, get_localised_images, get_image_decs, clear_folder, match, get_keypoints_xy

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

def get_img_data(model_img, img_file, points_3D, descs):
    img_data = np.empty([0, 134])
    for i in range(model_img.point3D_ids.shape[0]):  # can loop through descs or img_data.xys - same thing
        current_point3D_id = model_img.point3D_ids[i]

        if (current_point3D_id == -1):  # means feature (or keypoint) is unmatched
            matched = 0
        else:
            assert i in points_3D[current_point3D_id].point2D_idxs
            matched = 1

        desc = descs[i]  # np.uint8
        xy = model_img.xys[i]  # np.float64, same as xyz
        y = np.round(xy[1]).astype(int)
        x = np.round(xy[0]).astype(int)
        if (y >= img_file.shape[0] or x >= img_file.shape[1]):
            continue
        brg = img_file[y, x]  # opencv conventions
        # OpenCV uses BRG, not RGB
        blue = brg[0]
        green = brg[1]
        red = brg[2]

        # need to store in the same order as in the training data found in, getClassificationData()
        # SIFT + XY + RGB + Matched
        img_data = np.r_[img_data, np.append(desc, [xy[0], xy[1], blue, green, red, matched]).reshape([1, 134])]

    return img_data

def get_image_statistics(matchable_xy_descs, train_descriptors_gt, points3D_xyz_ids,
                         Ks, model_img, descs,
                         scale=1, camera_type=None, height=None,
                         width=None, train_descriptors_reference = None, points3D_xyz_ids_reference = None):
    # save feature reduction results
    percentage_reduction = 100 - (len(matchable_xy_descs) * 100 / len(descs))

    # Now start the matching
    queryDescriptors = matchable_xy_descs[:, 2:].astype(np.float32)
    keypoints_xy = matchable_xy_descs[:, 0:2]
    ratio_test_val = 1.0

    start = time.time()
    matches = match(queryDescriptors, train_descriptors_gt, keypoints_xy, points3D_xyz_ids, ratio_test_val, k=2)
    end = time.time()
    fm_time = end - start

    # To explain what is happening here:
    # For MnM I had to create a gt model that is build upon OpenCV SIFT and not COLMAP SIFT.
    # That gt model is only used to get the gt poses for MnM.
    # It is not fair to the other methods and MnM to measure feature time in that model.
    # So what I do I take the query descs of MnM which are the same size as the query descs of the other methods
    # for each image, (check format_data_for_match_no_match.py) and I match them to the original COLMAP train descriptors
    # I do not care about the matches in the latter case I just want to measure the feature time.
    # The matches used for pose estimation will have to come of course from the MnM gt model as I need to match Opencv SIFT
    # with OpenCV SIFT.
    # Placing this code here as it will overwrite fm_time
    if(train_descriptors_reference is not None and points3D_xyz_ids_reference is not None):
        start = time.time()
        _ = match(queryDescriptors, train_descriptors_reference, keypoints_xy, points3D_xyz_ids_reference, ratio_test_val, k=2)
        end = time.time()
        fm_time = end - start

    # get est pose at this point
    image_points = matches[:, 0:2]
    object_points = matches[:, 2:5]
    K = Ks[model_img.name]

    if(image_points.shape[0] < 4):
        return model_img.name

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
        return model_img.name

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

def get_MnM_pose_data(parameters, localised_query_images_mnm, mnm_model,
                      image_path, points3D_xyz_ids_mnm, Ks_mnm, thresholds_q, thresholds_t, scale=1, camera_type=None):

    train_descriptors_gt_mnm = np.load(parameters.avg_descs_gt_path_mnm).astype(np.float32)

    train_descriptors_gt = np.load(parameters.avg_descs_gt_path).astype(np.float32) # used later in get_image_statistics
    points3D_xyz_gt = read_points3d_default(parameters.gt_model_points3D_path) # used later in get_image_statistics
    points3D_xyz_id_gt = get_points3D_xyz_id(points3D_xyz_gt) # used later in get_image_statistics

    gt_db_mnm = COLMAPDatabase.connect(parameters.gt_db_path_mnm)
    total_fm_time = []  # feature matching time
    total_consencus_time = []  # RANSAC time
    percentage_reduction_total = []
    errors_rotation = []
    errors_translation = []
    est_poses = {}
    gt_poses = {}
    degenerate_images = []
    for model_img in tqdm(localised_query_images_mnm.values()):
        img_id = model_img.id
        kps_data = gt_db_mnm.execute(
            "SELECT rows, cols, data, octaves, angles, sizes, responses, greenIntensities, dominantOrientations, matched FROM keypoints WHERE image_id = ?",
            (img_id,)).fetchone()
        if (kps_data[9] == 99 or COLMAPDatabase.blob_to_array(kps_data[9], np.uint8).shape[0] == 0):
            # At this point for various reasons I did not add matched data to the database
            # for this specific localised image, so I will just skip it
            # Check format_data_for_match_no_match.py for more info
            # The second case happens when an image has keypoints but no image.xys for some reason (because of COLMAP most probably).
            print(f"Skipping image {model_img.name} ...")
            continue
        rows_no = kps_data[0]
        cols_no = kps_data[1]
        kps_xy = COLMAPDatabase.blob_to_array(kps_data[2], np.float32).reshape(rows_no, cols_no)  # (x,y) shape (rows_no, 2)
        octaves = COLMAPDatabase.blob_to_array(kps_data[3], np.uint8).reshape(rows_no, 1)  # octaves (rows_no, 1)
        angles = COLMAPDatabase.blob_to_array(kps_data[4], np.float32).reshape(rows_no, 1)
        sizes = COLMAPDatabase.blob_to_array(kps_data[5], np.float32).reshape(rows_no, 1)
        responses = COLMAPDatabase.blob_to_array(kps_data[6], np.float32).reshape(rows_no, 1)
        greenIntensities = COLMAPDatabase.blob_to_array(kps_data[7], np.uint8).reshape(rows_no, 1)
        dominantOrientations = COLMAPDatabase.blob_to_array(kps_data[8], np.uint8).reshape(rows_no, 1)
        matched = COLMAPDatabase.blob_to_array(kps_data[9], np.uint8).reshape(rows_no, 1)
        descs = get_image_decs(gt_db_mnm, img_id)
        # descs and image_data are in the same order
        image_data = np.c_[kps_xy, octaves, angles, sizes, responses, greenIntensities, dominantOrientations, matched]
        xy_descs = np.c_[kps_xy, descs]
        mnm_data = image_data[:, 0:8].astype(np.float32)

        _, y_pred_mnm = mnm_model.predict(mnm_data) #returns 1 or 0

        matchable_indices = np.where(y_pred_mnm == 1)[0]
        matchable_xy_descs = xy_descs[matchable_indices]

        # project on an image the matchable keypoints
        height, width = save_debug_image_simple_ml(os.path.join(image_path, model_img.name), kps_xy, matchable_xy_descs[:, 0:2],
                                   os.path.join(parameters.debug_images_ml_path, "mnm_"+model_img.name.replace("/", "_")))

        # data = {errors_rotation, errors_translation, total_fm_time, total_consencus_time, percentage_reduction_total, est_poses, gt_poses}
        data = get_image_statistics(matchable_xy_descs, train_descriptors_gt_mnm, points3D_xyz_ids_mnm, Ks_mnm, model_img, descs, scale=scale, camera_type=camera_type, height=height, width=width, train_descriptors_reference=train_descriptors_gt, points3D_xyz_ids_reference=points3D_xyz_id_gt)

        if type(data) == str: #returned failed image name
            print(f"No results for image: {data}")
            degenerate_images.append(data)
        else:
            total_fm_time.append(data["fm_time"])
            total_consencus_time.append(data["consensus_time"])
            percentage_reduction_total.append(data["percentage_reduction"])
            errors_rotation.append(data["error_rotation"])
            errors_translation.append(data["error_translation"])
            est_poses[model_img.name] = data["est_pose"]
            gt_poses[model_img.name] = data["gt_pose"]

    # at this point calculate the mAA
    mAA = pose_evaluate_generic_comparison_model_Maa(est_poses, gt_poses, thresholds_q, thresholds_t, scale=scale)
    results = {}
    results["percentage_reduction_total"] = percentage_reduction_total
    results["total_fm_time"] = total_fm_time
    results["total_consencus_time"] = total_consencus_time
    results["errors_rotation"] = errors_rotation
    results["errors_translation"] = errors_translation
    results["degenerate_images"] = degenerate_images

    return mAA, results

def get_NF_pose_data(parameters, localised_query_images, points_3D, nf_model,
                     image_path, points3D_xyz_ids, Ks, thresholds_q, thresholds_t, scale=1, camera_type=None):

    train_descriptors_gt = np.load(parameters.avg_descs_gt_path).astype(np.float32)
    gt_db = COLMAPDatabase.connect(parameters.gt_db_path)
    print("Evaluating gt localised images..")
    total_fm_time = []  # feature matching time
    total_consencus_time = []  # RANSAC time
    percentage_reduction_total = []
    errors_rotation = []
    errors_translation = []
    est_poses = {}
    gt_poses = {}
    degenerate_images = []
    for model_img in tqdm(localised_query_images.values()):
        img_id = model_img.id
        descs = get_image_decs(gt_db, img_id)
        # sanity checks
        assert (model_img.xys.shape[0] == model_img.point3D_ids.shape[0] == descs.shape[0])

        img_file = cv2.imread(os.path.join(image_path, model_img.name))  # at this point I am looking at gt images only
        img_data = get_img_data(model_img, img_file, points_3D, descs)

        # at this point we have all the data for the current image (we don't need the matched value here)
        # we are just predicting, we care about the predictions

        prediction_data = img_data[:, 0:133]
        y_pred = nf_model.predict(prediction_data, verbose = 0) #returns a value from (0,1)
        y_pred = np.where(y_pred >= 0.5, 1, 0)

        # re-organise the data so it is XY + SIFT
        descs = img_data[:, 0:128]
        xy = img_data[:, 128:130]
        xy_descs = np.c_[xy, descs]

        matchable_indices = np.where(y_pred == 1)[0]
        matchable_xy_descs = xy_descs[matchable_indices] #has to be XY + DESC

        kps_xy = get_keypoints_xy(gt_db, str(img_id)) #get the keypoints from the database

        # project on an image the matchable keypoints
        height, width = save_debug_image_simple_ml(os.path.join(image_path, model_img.name), kps_xy, matchable_xy_descs[:, 0:2],
                                   os.path.join(parameters.debug_images_ml_path, "nf_" + model_img.name.replace("/", "_")))

        data = get_image_statistics(matchable_xy_descs, train_descriptors_gt, points3D_xyz_ids, Ks, model_img, descs, scale=scale, camera_type=camera_type, height=height, width=width)

        if type(data) == str:  # returned failed image name
            print(f"No results for image: {data}")
            degenerate_images.append(data)
        else:
            total_fm_time.append(data["fm_time"])
            total_consencus_time.append(data["consensus_time"])
            percentage_reduction_total.append(data["percentage_reduction"])
            errors_rotation.append(data["error_rotation"])
            errors_translation.append(data["error_translation"])
            est_poses[model_img.name] = data["est_pose"]
            gt_poses[model_img.name] = data["gt_pose"]

    # at this point calculate the mAA
    mAA = pose_evaluate_generic_comparison_model_Maa(est_poses, gt_poses, thresholds_q, thresholds_t, scale=1)
    results = {}
    results["percentage_reduction_total"] = percentage_reduction_total
    results["total_fm_time"] = total_fm_time
    results["total_consencus_time"] = total_consencus_time
    results["errors_rotation"] = errors_rotation
    results["errors_translation"] = errors_translation
    results["degenerate_images"] = degenerate_images

    return mAA, results

def get_PM_pose_data(parameters, localised_query_images, points_3D, pm_model,
                     image_path, points3D_xyz_ids, Ks, thresholds_q, thresholds_t, scale=1, camera_type=None):

    train_descriptors_gt = np.load(parameters.avg_descs_gt_path).astype(np.float32)
    gt_db = COLMAPDatabase.connect(parameters.gt_db_path)
    print("Evaluating gt localised images..")
    total_fm_time = []  # feature matching time
    total_consencus_time = []  # RANSAC time
    percentage_reduction_total = []
    errors_rotation = []
    errors_translation = []
    est_poses = {}
    gt_poses = {}
    degenerate_images = []
    for model_img in tqdm(localised_query_images.values()):
        img_id = model_img.id
        descs = get_image_decs(gt_db, img_id)
        # sanity checks
        assert (model_img.xys.shape[0] == model_img.point3D_ids.shape[0] == descs.shape[0])

        img_file = cv2.imread(os.path.join(image_path, model_img.name))  # at this point I am looking at gt images only
        img_data = get_img_data(model_img, img_file, points_3D, descs)

        # at this point we have all the data for the current image (we don't need the matched value here)
        # we are just predicting, we care about the predictions

        prediction_data = img_data[:, 0:128] #only SIFT
        y_pred = pm_model.predict(prediction_data) #returns 1 or 0

        # re-organise the data so it is XY + SIFT
        descs = img_data[:, 0:128]
        xy = img_data[:, 128:130]
        xy_descs = np.c_[xy, descs]

        matchable_indices = np.where(y_pred == 1)[0]
        matchable_xy_descs = xy_descs[matchable_indices] #has to be XY + DESC

        kps_xy = get_keypoints_xy(gt_db, str(img_id)) #get the keypoints from the database

        # project on an image the matchable keypoints
        height, width = save_debug_image_simple_ml(os.path.join(image_path, model_img.name), kps_xy, matchable_xy_descs[:, 0:2],
                                   os.path.join(parameters.debug_images_ml_path, "pm_" + model_img.name.replace("/", "_")))

        # data = {errors_rotation, errors_translation, total_fm_time, total_consencus_time, percentage_reduction_total, est_poses, gt_poses}
        data = get_image_statistics(matchable_xy_descs, train_descriptors_gt, points3D_xyz_ids, Ks, model_img, descs, scale=scale, camera_type=camera_type, height=height, width=width)

        if type(data) == str:  # returned failed image name
            print(f"No results for image: {data}")
            degenerate_images.append(data)
        else:
            total_fm_time.append(data["fm_time"])
            total_consencus_time.append(data["consensus_time"])
            percentage_reduction_total.append(data["percentage_reduction"])
            errors_rotation.append(data["error_rotation"])
            errors_translation.append(data["error_translation"])
            est_poses[model_img.name] = data["est_pose"]
            gt_poses[model_img.name] = data["gt_pose"]

    # at this point calculate the mAA
    mAA = pose_evaluate_generic_comparison_model_Maa(est_poses, gt_poses, thresholds_q, thresholds_t, scale=1)
    results = {}
    results["percentage_reduction_total"] = percentage_reduction_total
    results["total_fm_time"] = total_fm_time
    results["total_consencus_time"] = total_consencus_time
    results["errors_rotation"] = errors_rotation
    results["errors_translation"] = errors_translation
    results["degenerate_images"] = degenerate_images

    return mAA, results

def get_localised_query_images_pose_data(parameters):
    all_query_images = read_images_binary(parameters.gt_model_images_path)  # only localised images (but from base,live,gt - we need only gt)
    all_query_images_names = load_images_from_text_file(parameters.gt_query_images_path)  # only gt images (all)

    localised_gt_images = get_localised_images(all_query_images_names, parameters.gt_model_images_path)  # only gt images (localised only)
    assert len(localised_gt_images) <= len(all_query_images_names)

    points3D_gt = read_points3d_default(parameters.gt_model_points3D_path)
    points3D_xyz_id_gt = get_points3D_xyz_id(points3D_gt)
    cameras_bin = read_cameras_binary(parameters.gt_model_cameras_path)
    Ks = get_intrinsics(all_query_images, cameras_bin)

    return localised_gt_images, Ks, points3D_xyz_id_gt

def get_localised_query_images_pose_data_mnm(parameters):
    all_query_images = read_images_binary(parameters.gt_model_images_path_mnm)  # only localised images (but from base,live,gt - we need only gt)
    all_query_images_names = load_images_from_text_file(parameters.query_gt_images_txt_path_mnm)  # only gt images (all)

    localised_qt_images = get_localised_images(all_query_images_names, parameters.gt_model_images_path_mnm)  # only gt images (localised only)
    assert len(localised_qt_images) <= len(all_query_images_names)

    points3D_gt = read_points3d_default(parameters.gt_model_points3D_path_mnm)
    points3D_xyz_id_gt = get_points3D_xyz_id(points3D_gt)
    cameras_bin = read_cameras_binary(parameters.gt_model_cameras_path_mnm)
    Ks = get_intrinsics(all_query_images, cameras_bin)

    return localised_qt_images, Ks, points3D_xyz_id_gt

def write_row(method_name, results_dict, mAA, writer):
    percentage_reduction_total = np.mean(results_dict["percentage_reduction_total"])
    total_fm_time = np.mean(results_dict["total_fm_time"])
    total_consencus_time = np.mean(results_dict["total_consencus_time"])
    errors_rotation = np.mean(results_dict["errors_rotation"])
    errors_translation = np.mean(results_dict["errors_translation"])
    degenerate_images_no = len(results_dict["degenerate_images"])

    writer.writerow(
        [method_name, f'{errors_translation}', f'{errors_rotation}', f'{percentage_reduction_total}', f'{total_fm_time}', f'{total_consencus_time}',
         f'{mAA[0]:}', f"{degenerate_images_no}"])

    print([method_name, f'{errors_translation:.3f}', f'{errors_rotation:.3f}', f'{percentage_reduction_total:.3f}', f'{total_fm_time:.3f}', f'{total_consencus_time:.3f}',
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

    # Doing MnM
    mnm_model_path = os.path.join(base_path, parameters.mnm_path, mnm_model_name)
    mnm_model = cv2.ml.RTrees_load(mnm_model_path)

    localised_query_images_mnm, Ks_mnm, points3D_xyz_ids_mnm = get_localised_query_images_pose_data_mnm(parameters)
    mAA_mnm, results_dict_mnm = get_MnM_pose_data(parameters, localised_query_images_mnm, mnm_model,
                                                  gt_image_path, points3D_xyz_ids_mnm, Ks_mnm,
                                                  thresholds_q, thresholds_t, scale=scale,
                                                  camera_type=camera_type)
    write_row("MnM", results_dict_mnm, mAA_mnm, writer)

    # Doing NF and PM

    # For NF (2023)
    nn_model_path = os.path.join(base_path, "ML_data", "classification_model")
    nf_model = keras.models.load_model(nn_model_path, compile=False)
    # For PM (2014)
    pm_model_path = os.path.join(base_path, parameters.predicting_matchability_comparison_data, pm_model_name)
    pm_model = load(pm_model_path)
    # needed for an assertion later
    gt_points_3D = read_points3d_default(parameters.gt_model_points3D_path)
    # Getting the COLMAP localised images used for NF and PM model
    localised_query_images, Ks, points3D_xyz_ids = get_localised_query_images_pose_data(parameters)

    mAA_nf, results_dict_nf = get_NF_pose_data(parameters, localised_query_images, gt_points_3D,
                                               nf_model, gt_image_path, points3D_xyz_ids,
                                               Ks, thresholds_q, thresholds_t, scale=scale,
                                               camera_type=camera_type)
    write_row("NF", results_dict_nf, mAA_nf, writer)

    mAA_pm, results_dict_pm = get_PM_pose_data(parameters, localised_query_images, gt_points_3D,
                                               pm_model, gt_image_path, points3D_xyz_ids,
                                               Ks, thresholds_q, thresholds_t, scale=scale,
                                               camera_type=camera_type)
    write_row("PM", results_dict_pm, mAA_pm, writer)

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
        pm_model_name = "rforest_1500.joblib"
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
        mnm_model_name = "trained_model_pairs_no_8000.xml"
        pm_model_name = "rforest_10500.joblib"
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
    pm_model_name = "rforest_3000.joblib"
    write_predictions(base_path, dataset, gt_image_path, thresholds_q, thresholds_t, writer, mnm_model_name, pm_model_name, ARCore=True)

root_path = "/media/iNicosiaData/engd_data/"
file_name = sys.argv[1]
result_file_output_path = os.path.join(root_path, file_name)

with open(result_file_output_path, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    pose_results_lamar(writer)
    pose_result_cmu(writer)
    pose_results_retail_shop(writer)

