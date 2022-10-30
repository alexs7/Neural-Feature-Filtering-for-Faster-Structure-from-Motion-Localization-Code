# This file was added to create 2D-3D matches for Match or No Match: Keypoint Filtering based on Matching Probability (2020) paper,
# for Predicting Matchability (2014) paper and for a vanillia RF model. Maybe I can add more models, fuck knows
import os
import subprocess
import time
from itertools import chain
from os.path import exists
import cv2
import numpy as np
from tqdm import tqdm
from query_image import get_image_id, get_keypoints_xy, get_keypoints_data, get_queryDescriptors
from save_2D_points import save_debug_image

def save_to_file_for_original_tool_prediction(all_descs, descs_path):
    with open(os.path.join(descs_path), 'w') as f:
        for desc in all_descs:
            row = ' '.join([str(num) for num in desc[0:128].astype(np.uint8)])
            f.write(f"{row}\n")

def feature_matcher_wrapper_generic_comparison_model_pm(base_path, comparison_data_path, model, db, query_images, trainDescriptors, points3D_xyz, ratio_test_val):
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    image_gt_dir = os.path.join(base_path, 'gt/images/')
    debug_images_path = os.path.join(comparison_data_path, "debug_images")
    images_percentage_reduction = {}
    images_matching_time = {}

    original_tool_path = "/home/Neural-Feature-Filtering-for-Faster-Structure-from-Motion-Localization-Code/code_to_compare/Predicting_Matchability/rforest"
    original_rforest_model_path = os.path.join(original_tool_path, "rforest.gz")
    original_test_descs_path = os.path.join(original_tool_path, "descs.txt")
    original_output_results_path = os.path.join(original_tool_path, "res.txt")
    original_tool_predict_command = os.path.join(original_tool_path, "./rforest")
    original_tool_test_time_path = "test_time.txt" #milliseconds

    if (exists(debug_images_path) == False):
        print("debug_images_path does not exist ! will create") #same images here will keep be overwritten no need to delete
        os.makedirs(debug_images_path, exist_ok=True)

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in tqdm(range(len(query_images))):
        total_time = 0
        query_image = query_images[i]
        image_gt_path = os.path.join(image_gt_dir, query_image)
        image_id = get_image_id(db,query_image)

        keypoints_xy = get_keypoints_xy(db, image_id)
        queryDescriptors = get_queryDescriptors(db, image_id)
        len_descs = queryDescriptors.shape[0]

        save_to_file_for_original_tool_prediction(queryDescriptors, original_test_descs_path)
        # rforest.exe -f rforest.gz -i desc.txt -o res.txt
        original_tool_exec = [original_tool_predict_command, "-f", original_rforest_model_path, "-i", original_test_descs_path, "-o", original_output_results_path]

        subprocess.check_call(original_tool_exec)
        elapsed_time = np.loadtxt(original_tool_test_time_path)/1000 #produced by the tool (in milliseconds)
        total_time += elapsed_time
        os.remove(original_tool_test_time_path)  #remove it now.

        # from tool doc: (the first and the second columns correspond to labels 0 and 1, respectively).
        predictions = np.loadtxt(original_output_results_path)
        predictions = predictions[:,1] #only care about the positive ones
        predictions[np.where(predictions > 0.5)] = 1 #this will set the values we want (> 0.5) to 1, the rest we don't care

        positive_samples_no = len(np.where(predictions == 1)[0])
        percentage_reduction = (100 - positive_samples_no * 100 / len_descs)

        save_debug_image(image_gt_path, keypoints_xy, keypoints_xy[predictions == 1], debug_images_path, query_image)

        # from now on I will be using the descs and keypoints that Predicting Matchability (2014) / MatchNoMatch 2020 deemed matchable
        queryDescriptors = queryDescriptors[predictions == 1]  # replacing queryDescriptors here so to keep code changes minimal
        keypoints_xy = keypoints_xy[predictions == 1]  # replacing keypoints_xy as they are mapped to queryDescriptors

        matcher = cv2.BFMatcher()  # cv2.FlannBasedMatcher(Parameters.index_params, Parameters.search_params) # or cv.BFMatcher()
        # Matching on trainDescriptors (remember these are the means of the 3D points)
        start = time.time()
        temp_matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)

        # output: idx1, idx2, lowes_distance (vectors of corresponding indexes in
        # m the closest, n is the second closest
        good_matches = []
        for m, n in temp_matches:
            assert(m.distance <= n.distance)
            # trainIdx is from 0 to no of points 3D (since each point 3D has a desc), so you can use it as an index here
            if (m.distance < ratio_test_val * n.distance): #and (score_m > score_n):
                if(m.queryIdx >= keypoints_xy.shape[0]): #keypoints_xy.shape[0] always same as classifier_predictions.shape[0]
                    raise Exception("m.queryIdx error!")
                if (m.trainIdx >= points3D_xyz.shape[0]):
                    raise Exception("m.trainIdx error!")
                # idx1.append(m.queryIdx)
                # idx2.append(m.trainIdx)
                scores = []
                xy2D = keypoints_xy[m.queryIdx, :].tolist()
                xyz3D = points3D_xyz[m.trainIdx, :].tolist()

                match_data = [xy2D, xyz3D, [m.distance, n.distance], scores]
                match_data = list(chain(*match_data))
                good_matches.append(match_data)

        # sanity check
        if (ratio_test_val == 1.0):
            if (len(good_matches) != len(temp_matches)):
                print(" Matches not equal, len(good_matches)= " + str(len(good_matches)) + " len(temp_matches)= " + str(len(temp_matches)))

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

    if (exists(debug_images_path) == False):
        print("debug_images_path does not exist ! will create") #same images here will keep be overwritten no need to delete
        os.makedirs(debug_images_path, exist_ok=True)

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

        save_debug_image(image_gt_path, keypoints_xy, keypoints_xy[predictions == 1], debug_images_path, query_image)

        # from now on I will be using the descs and keypoints that Predicting Matchability (2014) / MatchNoMatch 2020 deemed matchable
        queryDescriptors = queryDescriptors[predictions == 1]  # replacing queryDescriptors here so to keep code changes minimal
        keypoints_xy = keypoints_xy[predictions == 1]  # replacing keypoints_xy as they are mapped to queryDescriptors

        matcher = cv2.BFMatcher()  # cv2.FlannBasedMatcher(Parameters.index_params, Parameters.search_params) # or cv.BFMatcher()
        # Matching on trainDescriptors (remember these are the means of the 3D points)
        start = time.time()
        temp_matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)

        # output: idx1, idx2, lowes_distance (vectors of corresponding indexes in
        # m the closest, n is the second closest
        good_matches = []
        for m, n in temp_matches:
            assert(m.distance <= n.distance)
            # trainIdx is from 0 to no of points 3D (since each point 3D has a desc), so you can use it as an index here
            # 02/11/2022, added a = here to make is <=, because 1 fucking image returns exactly the same m and n distance.
            if (m.distance <= ratio_test_val * n.distance): #and (score_m > score_n):
                if(m.queryIdx >= keypoints_xy.shape[0]): #keypoints_xy.shape[0] always same as classifier_predictions.shape[0]
                    raise Exception("m.queryIdx error!")
                if (m.trainIdx >= points3D_xyz.shape[0]):
                    raise Exception("m.trainIdx error!")
                # idx1.append(m.queryIdx)
                # idx2.append(m.trainIdx)
                scores = []
                xy2D = keypoints_xy[m.queryIdx, :].tolist()
                xyz3D = points3D_xyz[m.trainIdx, :].tolist()

                match_data = [xy2D, xyz3D, [m.distance, n.distance], scores]
                match_data = list(chain(*match_data))
                good_matches.append(match_data)

        # sanity check
        if (ratio_test_val == 1.0):
            if (len(good_matches) != len(temp_matches)):
                print(" Matches not equal, len(good_matches)= " + str(len(good_matches)) + " len(temp_matches)= " + str(len(temp_matches)))

        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

        images_matching_time[query_image] = total_time
        images_percentage_reduction[query_image] = percentage_reduction
        matches[query_image] = np.array(good_matches)

    return matches, images_matching_time, images_percentage_reduction