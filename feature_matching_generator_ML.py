# This file is copied from my previous publication and using now with minor modification for the ML approach
#  such as not normalising the descriptors.
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

# indexing is the same as points3D indexing for trainDescriptors
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

# Will use raw descs not normalised
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
            random_idxs = np.random.choice(np.arange(len(queryDescriptors)), random_limit, replace=False)
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

    return matches, total_time

# This is used for benchmarking a ML model
# It will predict if a desc if matchable or not first, then pick "class_top" (sorted) matchable descs to do the feature matching
def feature_matcher_wrapper_model(classifier, regressor, db, query_images, trainDescriptors, points3D_xyz, ratio_test_val, verbose = False, points_scores_array=None, class_top = -1, regres_top = -1, pick_top_ones = False):
    assert(classifier is not None)
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    matches_sum = []
    total_time = 0
    matchable_threshold = 0.5
    keypoints_xy_descs_pred = np.empty([0, 131]) #(xy + SIFT + prediction val)

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in range(len(query_images)):
        query_image = query_images[i]
        if(verbose): print("Matching image " + str(i + 1) + "/" + str(len(query_images)) + ", " + query_image, end="\r")

        image_id = get_image_id(db,query_image)
        # keypoints data (first keypoint correspond to the first descriptor etc etc)
        keypoints_xy = get_keypoints_xy(db, image_id)
        queryDescriptors = get_queryDescriptors(db, image_id)

        start = time.time()
        classifier_predictions = classifier.predict_on_batch(queryDescriptors) #, use_multiprocessing=True, workers = 4)
        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

        # only keep matchable ones - discard the rest, NOTE: matchable_desc_indices sometimes can be less than 80!
        matchable_desc_indices = np.where(classifier_predictions > matchable_threshold)[0]  # matchable_desc_indices will index queryDescriptors/classifier_predictions
        matchable_desc_indices_length = matchable_desc_indices.shape[0]

        keypoints_xy = keypoints_xy[matchable_desc_indices]
        queryDescriptors = queryDescriptors[matchable_desc_indices]
        classifier_predictions = classifier_predictions[matchable_desc_indices]
        matchable_desc_indices = np.arange(matchable_desc_indices_length) # this used to index the above 'new' arrays after predictions

        # if pick_top_ones is set to True then it will sort by predicted value and pick the top ones (random_limit)
        # if not then it will pick random ones from a pool of only matchable descs
        if(pick_top_ones):
            start = time.time()
            classification_sorted_indices = classifier_predictions[:, 0].argsort()[::-1]
            end = time.time()
            elapsed_time = end - start
            total_time += elapsed_time

            keypoints_xy = keypoints_xy[classification_sorted_indices]
            queryDescriptors = queryDescriptors[classification_sorted_indices]
            # pick the top ones
            keypoints_xy = keypoints_xy[0:class_top, :]
            queryDescriptors = queryDescriptors[0:class_top, :]
        else:
            if(matchable_desc_indices_length > class_top): #if the network predicts more than 80 as matchable then pick random matchable 80.
                random_matchable_idx = np.random.choice(matchable_desc_indices, class_top, replace=False)
                keypoints_xy = keypoints_xy[random_matchable_idx]
                queryDescriptors = queryDescriptors[random_matchable_idx]
            else:
                keypoints_xy = keypoints_xy[matchable_desc_indices]
                queryDescriptors = queryDescriptors[matchable_desc_indices]

        # further processing if we want to use the regressor too
        if (regres_top != -1):
            start = time.time()
            regression_predictions = regressor.predict_on_batch(queryDescriptors)  # matchable only at this point
            regression_sorted_indices = regression_predictions[:, 0].argsort()[::-1]
            end = time.time()
            elapsed_time = end - start
            total_time += elapsed_time
            keypoints_xy = keypoints_xy[regression_sorted_indices]
            queryDescriptors = queryDescriptors[regression_sorted_indices]

            # pick the top ones
            keypoints_xy = keypoints_xy[0:regres_top, :]
            queryDescriptors = queryDescriptors[0:regres_top, :]

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
                if(m.queryIdx >= keypoints_xy.shape[0]): #keypoints_xy.shape[0] always same as classifier_predictions.shape[0]
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

    return matches, total_time
