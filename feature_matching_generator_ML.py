# This file is copied from my previous publication and using now with minor modification for the ML approach
#  such as not normalising the descriptors.
# creates 2d-3d matches data for ransac comparison

import time
from itertools import chain
import cv2
import numpy as np
import sys

# creates 2d-3d matches data for ransac comparison
def get_keypoints_xy(db, image_id):
    query_image_keypoints_data = db.execute("SELECT data FROM keypoints WHERE image_id = " + "'" + image_id + "'")
    query_image_keypoints_data = query_image_keypoints_data.fetchone()[0]
    query_image_keypoints_data_cols = db.execute("SELECT cols FROM keypoints WHERE image_id = " + "'" + image_id + "'")
    query_image_keypoints_data_cols = int(query_image_keypoints_data_cols.fetchone()[0])
    query_image_keypoints_data = db.blob_to_array(query_image_keypoints_data, np.float32)
    query_image_keypoints_data_rows = int(np.shape(query_image_keypoints_data)[0] / query_image_keypoints_data_cols)
    query_image_keypoints_data = query_image_keypoints_data.reshape(query_image_keypoints_data_rows, query_image_keypoints_data_cols)
    query_image_keypoints_data_xy = query_image_keypoints_data[:, 0:2]
    return query_image_keypoints_data_xy

# indexing is the same as points3D indexing for trainDescriptors - NOTE: This does not normalised the descriptors!
def get_queryDescriptors(db, image_id):
    query_image_descriptors_data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + image_id + "'")
    query_image_descriptors_data = query_image_descriptors_data.fetchone()[0]
    query_image_descriptors_data = db.blob_to_array(query_image_descriptors_data, np.uint8)
    descs_rows = int(np.shape(query_image_descriptors_data)[0] / 128)
    query_image_descriptors_data = query_image_descriptors_data.reshape([descs_rows, 128])
    queryDescriptors = query_image_descriptors_data.astype(np.float32)
    return queryDescriptors

def get_image_id(db, query_image):
    image_id = db.execute("SELECT image_id FROM images WHERE name = " + "'" + query_image + "'")
    image_id = str(image_id.fetchone()[0])
    return image_id

# Will use raw descs not normalised, used in prepare_comparison_data.py
def feature_matcher_wrapper_ml(db, query_images, trainDescriptors, points3D_xyz, ratio_test_val, verbose = False, points_scores_array=None, random_limit = -1):
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    matches_sum = []
    total_time = 0

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in range(len(query_images)):
        query_image = query_images[i]
        if(verbose): print("Matching image " + str(i + 1) + "/" + str(len(query_images)) + ", " + query_image, end="\r")

        image_id = get_image_id(db,query_image)
        # keypoints data (first keypoint correspond to the first descriptor etc etc)
        keypoints_xy = get_keypoints_xy(db, image_id)
        queryDescriptors = get_queryDescriptors(db, image_id)

        if(random_limit != -1):
            # len(queryDescriptors) or len(keypoints_xy) - should return the number or rows and be the same.
            len_descs = queryDescriptors.shape[0]
            percentage_num = int(len_descs * random_limit / 100)
            random_idxs = np.random.choice(np.arange(len_descs), percentage_num, replace=False)
            keypoints_xy = keypoints_xy[random_idxs]
            queryDescriptors = queryDescriptors[random_idxs]

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

                if (points_scores_array is not None):
                    for points_scores in points_scores_array:
                        scores.append(points_scores[0, m.trainIdx])
                        scores.append(points_scores[0, n.trainIdx])

                match_data = [xy2D, xyz3D, [m.distance, n.distance], scores]
                match_data = list(chain(*match_data))
                good_matches.append(match_data)

        # sanity check
        if(ratio_test_val == 1.0):
            assert len(good_matches) == len(temp_matches)

        matches[query_image] = np.array(good_matches)
        matches_sum.append(len(good_matches))

        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

    if(verbose):
        print()
        total_all_images = np.sum(matches_sum)
        print("Total matches: " + str(total_all_images) + ", no of images " + str(len(query_images)))
        matches_all_avg = total_all_images / len(matches_sum)
        print("Average matches per image: " + str(matches_all_avg) + ", no of images " + str(len(query_images)))

    total_avg_time = total_time / len(query_images)
    return matches, total_avg_time

# These will be used for benchmarking a ML model
# There are several cases here that wil be tested:
# classification - cl
# classification -> regression (latter, trained on matched only) - cl_rg
# regression (trained on all) - rg
# combined - cb
# For each of the above cases there is a seperate method, sadly loads of duplicate code at least it is clear to understand
# Depending on the case, It will predict if a desc if matchable or not first, then pick "class_top" (sorted) matchable descs to do the feature matching
def feature_matcher_wrapper_model_cl(db, query_images, trainDescriptors, points3D_xyz, ratio_test_val, classifier, top_no = None, verbose= True):
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    matches_sum = []
    total_time = 0
    matchable_threshold = 0.5
    percentage_reduction_total = 0

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in range(len(query_images)):
        query_image = query_images[i]
        # if(verbose):
        #     print("Matching image " + str(i + 1) + "/" + str(len(query_images)) + ", " + query_image)

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

        percentage_reduction_total = percentage_reduction_total + (100 - matchable_desc_indices_length * 100 / queryDescriptors.shape[0])

        keypoints_xy = keypoints_xy[matchable_desc_indices]
        queryDescriptors = queryDescriptors[matchable_desc_indices]
        classifier_predictions = classifier_predictions[matchable_desc_indices]

        if(top_no != None):
            percentage_num = int(len_descs * top_no / 100)
            start = time.time()
            classification_sorted_indices = classifier_predictions[:, 0].argsort()[::-1]
            end = time.time()
            elapsed_time = end - start
            total_time += elapsed_time
            keypoints_xy = keypoints_xy[classification_sorted_indices]
            queryDescriptors = queryDescriptors[classification_sorted_indices]
            # here I use the "percentage_num" value because as it was generated from the initial number of "queryDescriptors"
            keypoints_xy = keypoints_xy[0:percentage_num, :]
            queryDescriptors = queryDescriptors[0:percentage_num, :]

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

        matches[query_image] = np.array(good_matches)
        matches_sum.append(len(good_matches))

    if(verbose):
        print()
        total_all_images = np.sum(matches_sum)
        print("Total matches: " + str(total_all_images) + ", no of images " + str(len(query_images)))
        matches_all_avg = total_all_images / len(matches_sum)
        print("Average matches per image: " + str(matches_all_avg) + ", no of images " + str(len(query_images)))
        percentage_reduction_avg = percentage_reduction_total / len(query_images)
        print("Average matches percentage reduction per image (regardless of top_no): " + str(percentage_reduction_avg) + "%")

    total_avg_time = total_time / len(query_images)
    return matches, total_avg_time

def feature_matcher_wrapper_model_cl_rg(db, query_images, trainDescriptors, points3D_xyz, ratio_test_val, classifier, regressor, top_no = 10, verbose = True):
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    matches_sum = []
    total_time = 0
    matchable_threshold = 0.5

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in range(len(query_images)):
        query_image = query_images[i]
        # if(verbose):
        #     print("Matching image " + str(i + 1) + "/" + str(len(query_images)) + ", " + query_image)

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

        matches[query_image] = np.array(good_matches)
        matches_sum.append(len(good_matches))

    if(verbose):
        print()
        total_all_images = np.sum(matches_sum)
        print("Total matches: " + str(total_all_images) + ", no of images " + str(len(query_images)))
        matches_all_avg = total_all_images / len(matches_sum)
        print("Average matches per image: " + str(matches_all_avg) + ", no of images " + str(len(query_images)))

    total_avg_time = total_time / len(query_images)
    return matches, total_avg_time

def feature_matcher_wrapper_model_rg(db, query_images, trainDescriptors, points3D_xyz, ratio_test_val, regressor, top_no = 10, verbose = True):
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    matches_sum = []
    total_time = 0

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in range(len(query_images)):
        query_image = query_images[i]
        # if(verbose):
        #     print("Matching image " + str(i + 1) + "/" + str(len(query_images)) + ", " + query_image)

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

        matches[query_image] = np.array(good_matches)
        matches_sum.append(len(good_matches))

    if(verbose):
        print()
        total_all_images = np.sum(matches_sum)
        print("Total matches: " + str(total_all_images) + ", no of images " + str(len(query_images)))
        matches_all_avg = total_all_images / len(matches_sum)
        print("Average matches per image: " + str(matches_all_avg) + ", no of images " + str(len(query_images)))

    total_avg_time = total_time / len(query_images)
    return matches, total_avg_time

def feature_matcher_wrapper_model_cb(db, query_images, trainDescriptors, points3D_xyz, ratio_test_val, combined_model, top_no = 10, verbose = True):
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    matches_sum = []
    total_time = 0

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in range(len(query_images)):
        query_image = query_images[i]
        # if(verbose):
        #     print("Matching image " + str(i + 1) + "/" + str(len(query_images)) + ", " + query_image)

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

        matches[query_image] = np.array(good_matches)
        matches_sum.append(len(good_matches))

    if(verbose):
        print()
        total_all_images = np.sum(matches_sum)
        print("Total matches: " + str(total_all_images) + ", no of images " + str(len(query_images)))
        matches_all_avg = total_all_images / len(matches_sum)
        print("Average matches per image: " + str(matches_all_avg) + ", no of images " + str(len(query_images)))

    total_avg_time = total_time / len(query_images)
    return matches, total_avg_time