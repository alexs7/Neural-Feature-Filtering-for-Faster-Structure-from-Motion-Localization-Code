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

def compute_kp_scales(kp_data):
    scales = np.empty([kp_data.shape[0],1])
    for i in range(kp_data.shape[0]):
        a11 = kp_data[i][2]
        a12 = kp_data[i][3]
        a21 = kp_data[i][4]
        a22 = kp_data[i][5]
        scale = (np.sqrt(a11 * a11 + a21 * a21) + np.sqrt(a12 * a12 + a22 * a22)) / 2
        scales[i,:] = scale
    return scales

def compute_kp_orientations(kp_data):
    orientations = np.empty([kp_data.shape[0],1])
    for i in range(kp_data.shape[0]):
        a11 = kp_data[i][2]
        # a12 = kp_data[i][3]
        a21 = kp_data[i][4]
        # a22 = kp_data[i][5]
        orientation = np.arctan2(a21, a11)
        orientations[i,:] = orientation
    return orientations

def get_keypoints_meta_data(db, img_id):
    kp_data = db.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = " + "'" + str(img_id) + "'").fetchone()
    cols = kp_data[1]
    rows = kp_data[0]
    kp_data = COLMAPDatabase.blob_to_array(kp_data[2], np.float32)
    kp_data = kp_data.reshape([rows, cols])
    # Note: https://math.stackexchange.com/questions/13150/extracting-rotation-scale-values-from-2d-transformation-matrix
    # https://github.com/colmap/colmap/issues/1219
    # https://stackoverflow.com/questions/45159314/decompose-2d-transformation-matrix
    kp_scales = compute_kp_scales(kp_data)
    kp_orientations = compute_kp_orientations(kp_data)
    xy = kp_data[:,0:2]
    return np.c_[xy, kp_scales, kp_orientations]

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

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in tqdm(range(len(query_images))):
        total_time = 0
        query_image = query_images[i]
        image_gt_path = os.path.join(image_gt_dir, query_image)
        query_image_file = cv2.imread(image_gt_path)  # no need for cv2.COLOR_BGR2RGB here as G is in the middle anw
        image_width = query_image_file.shape[1]
        image_height = query_image_file.shape[0]
        image_id = get_image_id(db,query_image)

        # keypoints data (first keypoint correspond to the first descriptor etc etc)
        # both methods return xy
        keypoints_xy = get_keypoints_xy(db, image_id)
        # just removing outliers, keypoints that COLMAP detected outside the frame... (a COLMAP bug maybe)
        invalid_rows = np.argwhere((keypoints_xy[:, 0] > image_width) | (keypoints_xy[:, 1] > image_height))
        keypoints_xy = np.delete(keypoints_xy, invalid_rows, axis =0 )
        queryDescriptors = get_queryDescriptors(db, image_id)  # just to get their size
        queryDescriptors = np.delete(queryDescriptors, invalid_rows, axis =0 )
        assert(queryDescriptors.shape[0] == keypoints_xy.shape[0])
        len_descs = queryDescriptors.shape[0]

        keypoints_meta_data = get_keypoints_meta_data(db, image_id) #np.c_[xy, kp_scales, kp_orientations]
        scales = keypoints_meta_data[:,1]
        orientations = keypoints_meta_data[:,2]
        xs = keypoints_xy[:,0]
        ys = keypoints_xy[:,1]
        indxs = np.c_[np.round(ys), np.round(xs)].astype(np.int) #note the reverse here
        greenInt = query_image_file[indxs[:,0], indxs[:,1]][:,1]

        if(model_type == "MatchNoMatch"):
            # use extra data from MatchNoMatch paper
            test_data = np.c_[queryDescriptors, scales, orientations, xs, ys, greenInt].astype(np.float32)
        else:
            # use only SIFT
            test_data = queryDescriptors

        if (exists(comparison_data_path) == False):
            print("comparison_data_path does not exist")
            exit()

        os.makedirs(debug_images_path, exist_ok=True)

        start = time.time()
        # predictions_opencv = rtree_opencv.predict(test_data, cv2.ml.RAW_OUTPUT) #not working !
        predictions = model.predict(test_data)
        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

        positive_samples_no = len(np.where(predictions == 1)[0])
        percentage_reduction = (100 - positive_samples_no * 100 / len_descs)

        save_debug_image(image_gt_path, keypoints_xy, predictions, debug_images_path, query_image)

        # from now on I will be using the descs and keypoints that Predicting Matchability (2014) deemed matchable
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