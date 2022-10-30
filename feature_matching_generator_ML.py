# This file is copied from my previous publication and using now with minor modification for the ML approach
#  such as not normalising the descriptors.
# creates 2d-3d matches data for ransac comparison
import os
import time
from itertools import chain
from os.path import exists
import cv2
import numpy as np
import sys
from tqdm import tqdm
from query_image import get_image_id, get_keypoints_xy, get_queryDescriptors
from save_2D_points import save_debug_image

# Will use raw descs not normalised, used in prepare_comparison_data.py
def feature_matcher_wrapper_ml(base_path, db, query_images, trainDescriptors, points3D_xyz, ratio_test_val, output_path, random_limit = -1):
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    images_percentage_reduction = {}
    images_matching_time = {}
    image_gt_dir = os.path.join(base_path, 'gt/images/')

    debug_images_path = os.path.join(output_path, "debug_images")
    if (exists(debug_images_path) == False):
        print("debug_images_path does not exist! will create") #same images here will keep be overwritten no need to delete
        os.makedirs(debug_images_path, exist_ok=True)

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in tqdm(range(len(query_images))):
        total_time = 0
        query_image = query_images[i]
        image_gt_path = os.path.join(image_gt_dir, query_image)
        image_id = get_image_id(db,query_image)
        # keypoints data (first keypoint correspond to the first descriptor etc etc)
        keypoints_xy = get_keypoints_xy(db, image_id)
        queryDescriptors = get_queryDescriptors(db, image_id)

        if(random_limit != -1):
            # len(queryDescriptors) or len(keypoints_xy) - should return the number or rows and be the same.
            len_descs = keypoints_xy.shape[0]
            percentage_num = int(len_descs * random_limit / 100)
            keypoints_idxs = np.arange(len_descs)
            np.random.shuffle(keypoints_idxs)
            rnd_idx = keypoints_idxs[:percentage_num]
            # save here, before keypoints_xy get overwritten
            save_debug_image(image_gt_path, keypoints_xy, keypoints_xy[rnd_idx], debug_images_path, query_image)  # random
            keypoints_xy = keypoints_xy[rnd_idx]
            queryDescriptors = queryDescriptors[rnd_idx]
        else:
            save_debug_image(image_gt_path, keypoints_xy, keypoints_xy, debug_images_path, query_image) #pass keypoints_xy 2 times, here as no predictions

        matcher = cv2.BFMatcher()  # cv2.FlannBasedMatcher(Parameters.index_params, Parameters.search_params) # or cv.BFMatcher()
        # Matching on trainDescriptors (remember these are the means of the 3D points)
        start = time.time()
        temp_matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)

        # output: idx1, idx2, lowes_distance (vectors of corresponding indexes in
        # m the closest, n is the second closest
        good_matches = []
        for m, n in temp_matches: # TODO: maybe consider what you have at this point? and add it to the if condition ?
            assert(m.distance <=  n.distance)
            # trainIdx is from 0 to no of points 3D (since each point 3D has a desc), so you can use it as an index here
            if (m.distance < ratio_test_val * n.distance): #and (score_m > score_n):
                if(m.queryIdx >= keypoints_xy.shape[0]): #keypoints_xy.shape[0] always same as queryDescriptors.shape[0]
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
        if(ratio_test_val == 1.0):
            assert len(good_matches) == len(temp_matches)

        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

        if (random_limit != -1):
            percentage_reduction = 100 - random_limit  # random %
        else:
            percentage_reduction = 0  # all baseline

        images_matching_time[query_image] = total_time
        images_percentage_reduction[query_image] = percentage_reduction
        matches[query_image] = np.array(good_matches)

    return matches, images_matching_time, images_percentage_reduction

# These will be used for benchmarking a ML model
# There are several cases here that wil be tested:
# classification - cl
# classification -> regression (latter, trained on matched only) - cl_rg
# regression (trained on all) - rg
# combined - cb
# For each of the above cases there is a seperate method, sadly loads of duplicate code at least it is clear to understand
# Depending on the case, It will predict if a desc if matchable or not first, then pick "class_top" (sorted) matchable descs to do the feature matching
def feature_matcher_wrapper_model_cl(base_path, ml_path, ml_model_idx, db, query_images, trainDescriptors, points3D_xyz, ratio_test_val, classifier, top_no = None):
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    matchable_threshold = 0.5
    images_percentage_reduction = {}
    images_matching_time = {}
    image_gt_dir = os.path.join(base_path, 'gt/images/')

    debug_images_path = os.path.join(ml_path, f"debug_images_model_{ml_model_idx}")
    if (exists(debug_images_path) == False):
        print("debug_images_path does not exist! will create")
        os.makedirs(debug_images_path, exist_ok=True)

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in tqdm(range(len(query_images))):
        total_time = 0
        query_image = query_images[i]
        image_gt_path = os.path.join(image_gt_dir, query_image)
        image_id = get_image_id(db,query_image)
        # keypoints data (first keypoint correspond to the first descriptor etc etc)
        keypoints_xy = get_keypoints_xy(db, image_id)
        queryDescriptors = get_queryDescriptors(db, image_id)
        len_descs = queryDescriptors.shape[0]

        start = time.time()
        classifier_predictions = classifier.predict_on_batch(queryDescriptors) #, use_multiprocessing=True, workers = 4)
        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

        # only keep matchable ones - discard the rest, NOTE: matchable_desc_indices sometimes can be less than the 10% or whatever percentage!
        matchable_desc_indices = np.where(classifier_predictions > matchable_threshold)[0]  # matchable_desc_indices will index queryDescriptors/classifier_predictions
        matchable_desc_indices_length = matchable_desc_indices.shape[0]

        percentage_reduction = (100 - matchable_desc_indices_length * 100 / queryDescriptors.shape[0])

        keypoints_xy = keypoints_xy[matchable_desc_indices]
        queryDescriptors = queryDescriptors[matchable_desc_indices]
        classifier_predictions = classifier_predictions[matchable_desc_indices]

        if(top_no != None):
            percentage_num = int(len_descs * top_no / 100)
            start = time.time()
            classification_sorted_indices = classifier_predictions[:, 0].argsort()[::-1]
            end = time.time()
            elapsed_time = end - start
            total_time += elapsed_time #TODO: Maybe remove time addition 15/09/2022 ?
            keypoints_xy = keypoints_xy[classification_sorted_indices]
            queryDescriptors = queryDescriptors[classification_sorted_indices]
            # here I use the "percentage_num" value because as it was generated from the initial number of "queryDescriptors"
            keypoints_xy = keypoints_xy[0:percentage_num, :]
            queryDescriptors = queryDescriptors[0:percentage_num, :]

        original_keypoints = get_keypoints_xy(db, image_id)
        save_debug_image(image_gt_path, original_keypoints, keypoints_xy, debug_images_path, query_image) #keypoints_xy = predicted ones

        matcher = cv2.BFMatcher()  # cv2.FlannBasedMatcher(Parameters.index_params, Parameters.search_params) # or cv.BFMatcher()
        # Matching on trainDescriptors (remember these are the means of the 3D points)
        start = time.time()
        temp_matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)

        # output: idx1, idx2, lowes_distance (vectors of corresponding indexes in
        # m the closest, n is the second closest
        good_matches = []
        for m, n in temp_matches: # TODO: maybe consider what you have at this point? and add it to the if condition ?
            assert(m.distance <= n.distance) #TODO: maybe count how many pass the ratio test VS how many they dont without the NN ?
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

                # TODO: add a flag and predict a score for each match to use later in PROSAC
                match_data = [xy2D, xyz3D, [m.distance, n.distance], scores]
                match_data = list(chain(*match_data))
                good_matches.append(match_data)

        # sanity check
        if (ratio_test_val == 1.0):
            assert len(good_matches) == len(temp_matches)

        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

        images_matching_time[query_image] = total_time
        images_percentage_reduction[query_image] = percentage_reduction
        matches[query_image] = np.array(good_matches)

    return matches, images_matching_time, images_percentage_reduction

def feature_matcher_wrapper_model_cl_rg(base_path, ml_path, ml_model_idx, db, query_images, trainDescriptors, points3D_xyz, ratio_test_val, classifier, regressor, top_no = 10):
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    matchable_threshold = 0.5
    images_percentage_reduction = {}
    images_matching_time = {}
    image_gt_dir = os.path.join(base_path, 'gt/images/')

    debug_images_path = os.path.join(ml_path, f"debug_images_model_{ml_model_idx}")
    if (exists(debug_images_path) == False):
        print("debug_images_path does not exist! will create")
        os.makedirs(debug_images_path, exist_ok=True)

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in tqdm(range(len(query_images))):
        total_time = 0
        query_image = query_images[i]
        image_gt_path = os.path.join(image_gt_dir, query_image)
        image_id = get_image_id(db,query_image)
        # keypoints data (first keypoint correspond to the first descriptor etc etc)
        keypoints_xy = get_keypoints_xy(db, image_id)
        queryDescriptors = get_queryDescriptors(db, image_id)
        len_descs = queryDescriptors.shape[0]
        percentage_num = int(len_descs * top_no / 100)

        start = time.time()
        classifier_predictions = classifier.predict_on_batch(queryDescriptors) #, use_multiprocessing=True, workers = 4)
        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

        # only keep matchable ones - discard the rest, NOTE: matchable_desc_indices sometimes can be less than 80!
        matchable_desc_indices = np.where(classifier_predictions > matchable_threshold)[0]  # matchable_desc_indices will index queryDescriptors/classifier_predictions

        keypoints_xy = keypoints_xy[matchable_desc_indices]
        queryDescriptors = queryDescriptors[matchable_desc_indices]

        start = time.time()
        regression_predictions = regressor.predict_on_batch(queryDescriptors)  # matchable only at this point
        regression_sorted_indices = regression_predictions[:, 0].argsort()[::-1]
        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time
        keypoints_xy = keypoints_xy[regression_sorted_indices]
        queryDescriptors = queryDescriptors[regression_sorted_indices]
        sorted_regression_predictions = regression_predictions[regression_sorted_indices]

        # pick the top ones after regression predictions, using "percentage_num"
        # all have the same order below
        keypoints_xy = keypoints_xy[0:percentage_num, :]
        queryDescriptors = queryDescriptors[0:percentage_num, :]
        sorted_regression_predictions = sorted_regression_predictions[0:percentage_num, :]

        original_keypoints = get_keypoints_xy(db, image_id)
        save_debug_image(image_gt_path, original_keypoints, keypoints_xy, debug_images_path, query_image)  # keypoints_xy = predicted ones

        matcher = cv2.BFMatcher()  # cv2.FlannBasedMatcher(Parameters.index_params, Parameters.search_params) # or cv.BFMatcher()
        # Matching on trainDescriptors (remember these are the means of the 3D points)
        start = time.time()
        temp_matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)

        # output: idx1, idx2, lowes_distance (vectors of corresponding indexes in
        # m the closest, n is the second closest
        good_matches = []
        for m, n in temp_matches: # TODO: maybe consider what you have at this point? and add it to the if condition ?
            assert(m.distance <= n.distance) #TODO: maybe count how many pass the ratio test VS how many they dont without the NN ?
            # trainIdx is from 0 to no of points 3D (since each point 3D has a desc), so you can use it as an index here
            if (m.distance < ratio_test_val * n.distance): #and (score_m > score_n):
                if(m.queryIdx >= keypoints_xy.shape[0]): #keypoints_xy.shape[0] always same as classifier_predictions.shape[0]
                    raise Exception("m.queryIdx error!")
                if (m.trainIdx >= points3D_xyz.shape[0]):
                    raise Exception("m.trainIdx error!")
                # idx1.append(m.queryIdx)
                # idx2.append(m.trainIdx)
                xy2D = keypoints_xy[m.queryIdx, :].tolist()
                xyz3D = points3D_xyz[m.trainIdx, :].tolist()
                regression_prediction = sorted_regression_predictions[m.queryIdx, 0]

                # TODO: add a flag and predict a score for each match to use later in PROSAC
                match_data = [xy2D, xyz3D, [m.distance, n.distance], [regression_prediction]]
                match_data = list(chain(*match_data))
                good_matches.append(match_data)

        # sanity check
        if (ratio_test_val == 1.0):
            assert len(good_matches) == len(temp_matches)

        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

        images_matching_time[query_image] = total_time
        images_percentage_reduction[query_image] = 90 #this is just going to be 90% as we only pick the 10%
        matches[query_image] = np.array(good_matches)

    return matches, images_matching_time, images_percentage_reduction

def feature_matcher_wrapper_model_rg(base_path, ml_path, ml_model_idx, db, query_images, trainDescriptors, points3D_xyz, ratio_test_val, regressor, top_no = 10):
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    images_percentage_reduction = {}
    images_matching_time = {}
    image_gt_dir = os.path.join(base_path, 'gt/images/')

    debug_images_path = os.path.join(ml_path, f"debug_images_model_{ml_model_idx}")
    if (exists(debug_images_path) == False):
        print("debug_images_path does not exist! will create")
        os.makedirs(debug_images_path, exist_ok=True)

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in tqdm(range(len(query_images))):
        total_time = 0
        query_image = query_images[i]
        image_gt_path = os.path.join(image_gt_dir, query_image)
        image_id = get_image_id(db,query_image)
        # keypoints data (first keypoint correspond to the first descriptor etc etc)
        keypoints_xy = get_keypoints_xy(db, image_id)
        queryDescriptors = get_queryDescriptors(db, image_id)
        len_descs = queryDescriptors.shape[0]
        percentage_num = int(len_descs * top_no / 100)

        start = time.time()
        regression_predictions = regressor.predict_on_batch(queryDescriptors)  # matchable only at this point
        regression_sorted_indices = regression_predictions[:, 0].argsort()[::-1]
        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time
        keypoints_xy = keypoints_xy[regression_sorted_indices]
        queryDescriptors = queryDescriptors[regression_sorted_indices]
        sorted_regression_predictions = regression_predictions[regression_sorted_indices]

        # pick the top ones after regression predictions, using "percentage_num"
        # all have the same order below
        keypoints_xy = keypoints_xy[0:percentage_num, :]
        queryDescriptors = queryDescriptors[0:percentage_num, :]
        sorted_regression_predictions = sorted_regression_predictions[0:percentage_num, :]

        original_keypoints = get_keypoints_xy(db, image_id)
        save_debug_image(image_gt_path, original_keypoints, keypoints_xy, debug_images_path, query_image)  # keypoints_xy = predicted ones

        matcher = cv2.BFMatcher()  # cv2.FlannBasedMatcher(Parameters.index_params, Parameters.search_params) # or cv.BFMatcher()
        # Matching on trainDescriptors (remember these are the means of the 3D points)
        start = time.time()
        temp_matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)

        # output: idx1, idx2, lowes_distance (vectors of corresponding indexes in
        # m the closest, n is the second closest
        good_matches = []
        for m, n in temp_matches: # TODO: maybe consider what you have at this point? and add it to the if condition ?
            assert(m.distance <= n.distance) #TODO: maybe count how many pass the ratio test VS how many they dont without the NN ?
            # trainIdx is from 0 to no of points 3D (since each point 3D has a desc), so you can use it as an index here
            if (m.distance < ratio_test_val * n.distance): #and (score_m > score_n):
                if(m.queryIdx >= keypoints_xy.shape[0]): #keypoints_xy.shape[0] always same as classifier_predictions.shape[0]
                    raise Exception("m.queryIdx error!")
                if (m.trainIdx >= points3D_xyz.shape[0]):
                    raise Exception("m.trainIdx error!")
                # idx1.append(m.queryIdx)
                # idx2.append(m.trainIdx)
                xy2D = keypoints_xy[m.queryIdx, :].tolist()
                xyz3D = points3D_xyz[m.trainIdx, :].tolist()
                regression_prediction = sorted_regression_predictions[m.queryIdx, 0]

                # TODO: add a flag and predict a score for each match to use later in PROSAC
                match_data = [xy2D, xyz3D, [m.distance, n.distance], [regression_prediction]]
                match_data = list(chain(*match_data))
                good_matches.append(match_data)

        # sanity check
        if (ratio_test_val == 1.0):
            assert len(good_matches) == len(temp_matches)

        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

        images_matching_time[query_image] = total_time
        images_percentage_reduction[query_image] = 90  # this is just going to be 90% as we only pick the 10%
        matches[query_image] = np.array(good_matches)

    return matches, images_matching_time, images_percentage_reduction

def feature_matcher_wrapper_model_cb(base_path, ml_path, ml_model_idx, db, query_images, trainDescriptors, points3D_xyz, ratio_test_val, combined_model, top_no = 10):
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    images_percentage_reduction = {}
    images_matching_time = {}
    image_gt_dir = os.path.join(base_path, 'gt/images/')

    debug_images_path = os.path.join(ml_path, f"debug_images_model_{ml_model_idx}")
    if (exists(debug_images_path) == False):
        print("debug_images_path does not exist! will create")
        os.makedirs(debug_images_path, exist_ok=True)

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in tqdm(range(len(query_images))):
        total_time = 0
        query_image = query_images[i]
        image_gt_path = os.path.join(image_gt_dir, query_image)
        image_id = get_image_id(db,query_image)
        # keypoints data (first keypoint correspond to the first descriptor etc etc)
        keypoints_xy = get_keypoints_xy(db, image_id)
        queryDescriptors = get_queryDescriptors(db, image_id)
        len_descs = queryDescriptors.shape[0]
        percentage_num = int(len_descs * top_no / 100)

        start = time.time()
        regression_predictions = combined_model.predict_on_batch(queryDescriptors)  # matchable only at this point
        regression_predictions = np.add(regression_predictions[0], regression_predictions[1]) #add outputs
        regression_sorted_indices = regression_predictions[:, 0].argsort()[::-1]
        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

        keypoints_xy = keypoints_xy[regression_sorted_indices]
        queryDescriptors = queryDescriptors[regression_sorted_indices]
        sorted_regression_predictions = regression_predictions[regression_sorted_indices]

        # pick the top ones after combined_model predictions, using "percentage_num"
        # all have the same order below
        keypoints_xy = keypoints_xy[0:percentage_num, :]
        queryDescriptors = queryDescriptors[0:percentage_num, :]
        sorted_regression_predictions = sorted_regression_predictions[0:percentage_num, :]

        original_keypoints = get_keypoints_xy(db, image_id)
        save_debug_image(image_gt_path, original_keypoints, keypoints_xy, debug_images_path, query_image)  # keypoints_xy = predicted ones

        matcher = cv2.BFMatcher()  # cv2.FlannBasedMatcher(Parameters.index_params, Parameters.search_params) # or cv.BFMatcher()
        # Matching on trainDescriptors (remember these are the means of the 3D points)
        start = time.time()
        temp_matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)

        # output: idx1, idx2, lowes_distance (vectors of corresponding indexes in
        # m the closest, n is the second closest
        good_matches = []
        for m, n in temp_matches: # TODO: maybe consider what you have at this point? and add it to the if condition ?
            assert(m.distance <= n.distance) #TODO: maybe count how many pass the ratio test VS how many they dont without the NN ?
            # trainIdx is from 0 to no of points 3D (since each point 3D has a desc), so you can use it as an index here
            # if ratio_test_val == 1 was added to just add all the temp matches to good matches - skip the check below
            if (m.distance < ratio_test_val * n.distance or ratio_test_val == 1.0): #and (score_m > score_n):
                if(m.queryIdx >= keypoints_xy.shape[0]): #keypoints_xy.shape[0] always same as classifier_predictions.shape[0]
                    raise Exception("m.queryIdx error!")
                if (m.trainIdx >= points3D_xyz.shape[0]):
                    raise Exception("m.trainIdx error!")
                # idx1.append(m.queryIdx)
                # idx2.append(m.trainIdx)
                xy2D = keypoints_xy[m.queryIdx, :].tolist()
                xyz3D = points3D_xyz[m.trainIdx, :].tolist()
                combined_model_prediction = sorted_regression_predictions[m.queryIdx, 0]

                # TODO: add a flag and predict a score for each match to use later in PROSAC
                match_data = [xy2D, xyz3D, [m.distance, n.distance], [combined_model_prediction]]
                match_data = list(chain(*match_data))
                good_matches.append(match_data)

        # sanity check
        if (ratio_test_val == 1.0):
            assert len(good_matches) == len(temp_matches)

        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

        images_matching_time[query_image] = total_time
        images_percentage_reduction[query_image] = 90 #this is just going to be 90% as we only pick the 10%
        matches[query_image] = np.array(good_matches)

    return matches, images_matching_time, images_percentage_reduction