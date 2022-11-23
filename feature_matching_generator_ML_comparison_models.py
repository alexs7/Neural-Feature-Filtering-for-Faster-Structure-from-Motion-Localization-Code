# This file was added to create 2D-3D matches for Match or No Match: Keypoint Filtering based on Matching Probability (2020) paper and
# for Predicting Matchability (2014) paper. Maybe I can add more models, fuck knows
import os
import time
from os.path import exists
import cv2
import numpy as np
from tqdm import tqdm
from query_image import get_image_id, get_keypoints_xy, get_keypoints_data, get_queryDescriptors, match

def feature_matcher_wrapper_generic_comparison_model_pm(base_path, comparison_data_path, model, db, query_images, trainDescriptors, points3D_xyz, ratio_test_val):
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    debug_images_path = os.path.join(comparison_data_path, "debug_images")
    images_percentage_reduction = {}
    images_matching_time = {}

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in tqdm(range(len(query_images))):
        total_time = 0
        query_image = query_images[i]
        image_id = get_image_id(db,query_image)

        keypoints_xy = get_keypoints_xy(db, image_id)
        queryDescriptors = get_queryDescriptors(db, image_id)
        len_descs = queryDescriptors.shape[0]

        start = time.time()
        predictions = model.predict(queryDescriptors)
        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

        positive_samples_no = len(np.where(predictions == 1)[0])
        percentage_reduction = (100 - positive_samples_no * 100 / len_descs)

        # from now on I will be using the descs and keypoints that Predicting Matchability (2014) / MatchNoMatch 2020 deemed matchable
        queryDescriptors = queryDescriptors[predictions == 1]  # replacing queryDescriptors here so to keep code changes minimal
        keypoints_xy = keypoints_xy[predictions == 1]  # replacing keypoints_xy as they are mapped to queryDescriptors

        start = time.time()
        good_matches = match(queryDescriptors, trainDescriptors, keypoints_xy, points3D_xyz, ratio_test_val, k=2)
        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

        images_matching_time[query_image] = total_time
        images_percentage_reduction[query_image] = percentage_reduction
        matches[query_image] = np.array(good_matches)

    return matches, images_matching_time, images_percentage_reduction

# Two methods below are duplicated of each other - for now it is easier to implement. TODO: Refactor (27/10/2022)
def feature_matcher_wrapper_generic_comparison_model_mnm(base_path, comparison_data_path, model, db, query_images, trainDescriptors, points3D_xyz, ratio_test_val):
    # NOTE: This method uses OpenCV descriptors saved, in a seperate folder
    matches = {}     # create image_name <-> matches, dict - easier to work with
    image_gt_dir = os.path.join(base_path, 'gt/images/')
    debug_images_path = os.path.join(comparison_data_path, "debug_images")
    images_percentage_reduction = {}
    images_matching_time = {}

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in tqdm(range(len(query_images))):
        total_time = 0
        query_image = query_images[i]
        image_gt_path = os.path.join(image_gt_dir, query_image)
        query_image_file = cv2.imread(image_gt_path)  # no need for cv2.COLOR_BGR2RGB here as G is in the middle anw
        image_id = get_image_id(db,query_image)

        queryDescriptors = get_queryDescriptors(db, image_id)
        keypoints_data = get_keypoints_data(db, image_id, query_image_file)
        len_descs = keypoints_data.shape[0]
        keypoints_xy = keypoints_data[:,0:2]

        # just a couple of checks
        assert len(keypoints_xy) == len(queryDescriptors)
        assert len(keypoints_data) == len(keypoints_xy)

        start = time.time()
        _ , predictions = model.predict(keypoints_data) #array of arrays
        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

        predictions = np.array(predictions.ravel()).astype(np.uint8)
        positive_samples_no = len(np.where(predictions == 1)[0])
        percentage_reduction = (100 - positive_samples_no * 100 / len_descs)

        # from now on I will be using the descs and keypoints that Predicting Matchability (2014) / MatchNoMatch 2020 deemed matchable
        queryDescriptors = queryDescriptors[predictions == 1]  # replacing queryDescriptors here so to keep code changes minimal
        keypoints_xy = keypoints_xy[predictions == 1]  # replacing keypoints_xy as they are mapped to queryDescriptors

        start = time.time()
        good_matches = match(queryDescriptors, trainDescriptors, keypoints_xy, points3D_xyz, ratio_test_val, k=2)
        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

        images_matching_time[query_image] = total_time
        images_percentage_reduction[query_image] = percentage_reduction
        matches[query_image] = np.array(good_matches)

    return matches, images_matching_time, images_percentage_reduction