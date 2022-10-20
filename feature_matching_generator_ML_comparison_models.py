# This file was added to create 2D-3D matches for Match or No Match: Keypoint Filtering based on Matching Probability (2020) paper,
# for Predicting Matchability (2014) paper and for a vanillia RF model. Maybe I can add more models, fuck knows
import os
import time
from itertools import chain
from os.path import exists
import cv2
import numpy as np
from tqdm import tqdm

from database import COLMAPDatabase
from save_2D_points import save_debug_image

def get_keypoints_data(db, img_id, model_type=None):
    if(model_type=="MatchNoMatch"):
        # it is a row, with many keypoints (blob)
        kp_db_row = db.execute("SELECT rows, cols, data, dominantOrientations FROM keypoints WHERE image_id = " + "'" + str(img_id) + "'").fetchone()
        cols = kp_db_row[1]
        rows = kp_db_row[0]
        # x, y, octave, angle, size, response
        kp_data = COLMAPDatabase.blob_to_array(kp_db_row[2], np.float32)
        kp_data = kp_data.reshape([rows, cols])
        dominantOrientations = COLMAPDatabase.blob_to_array(kp_db_row[3], np.uint8)
        dominantOrientations = dominantOrientations.reshape([rows, 1])
        # xs, ys, octaves, angles, sizes, responses, dominantOrientations
        return np.c_[kp_data, dominantOrientations]
    else:
        # it is a row, with many keypoints (blob)
        kp_db_row = db.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = " + "'" + str(img_id) + "'").fetchone()
        cols = kp_db_row[1]
        rows = kp_db_row[0]
        kp_data = COLMAPDatabase.blob_to_array(kp_db_row[2], np.float32)
        kp_data = kp_data.reshape([rows, cols])
        # I only care about x and y here
        return kp_data

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

def feature_matcher_wrapper_generic_comparison_model(base_path, comparison_data_path, model, db, query_images, trainDescriptors, points3D_xyz, ratio_test_val, model_type=None):
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    image_gt_dir = os.path.join(base_path, 'gt/images/')
    debug_images_path = os.path.join(comparison_data_path, "debug_images")
    images_percentage_reduction = {}
    images_matching_time = {}

    if (exists(comparison_data_path) == False):
        print("comparison_data_path does not exist ! will create")
        os.makedirs(debug_images_path, exist_ok=True)

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in tqdm(range(len(query_images))):
        total_time = 0
        query_image = query_images[i]
        image_gt_path = os.path.join(image_gt_dir, query_image)
        query_image_file = cv2.imread(image_gt_path)  # no need for cv2.COLOR_BGR2RGB here as G is in the middle anw
        image_width = query_image_file.shape[1]
        image_height = query_image_file.shape[0]
        image_id = get_image_id(db,query_image)

        queryDescriptors = get_queryDescriptors(db, image_id)
        keypoints_data = get_keypoints_data(db, image_id, model_type)
        xs = keypoints_data[:, 0]
        ys = keypoints_data[:, 1]
        indxs = np.c_[np.round(ys), np.round(xs)].astype(np.int)  # note the reverse here
        greenInt = query_image_file[indxs[:, 0], indxs[:, 1]][:, 1]
        test_data = np.c_[keypoints_data, greenInt, queryDescriptors]
        len_descs = queryDescriptors.shape[0]

        invalid_rows = np.argwhere((test_data[:, 0] > image_width) | (test_data[:, 1] > image_height))

        test_data = np.delete(queryDescriptors, invalid_rows, axis=0)

        if(model_type == "MatchNoMatch"):
            query_data = test_data[:, 0:8] # x, y, octave, angle, size, response, dominantOrientation, green_intensity
        else:
            query_data = test_data[:, :128] #sift

        start = time.time()
        # predictions_opencv = rtree_opencv.predict(test_data, cv2.ml.RAW_OUTPUT) #not working !
        predictions = model.predict(query_data)
        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

        positive_samples_no = len(np.where(predictions == 1)[0])
        percentage_reduction = (100 - positive_samples_no * 100 / len_descs)
        assert(predictions.shape[0] == test_data.shape[0])

        predicted_keypoint_xy = test_data[predictions == 1] #predictions and test_data must have the same rows

        keypoints_xy = test_data[:,0:2]
        save_debug_image(image_gt_path, keypoints_xy, predicted_keypoint_xy, debug_images_path, query_image)

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