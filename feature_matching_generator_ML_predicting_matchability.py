# This file is copied from my previous publication and using now with minor modification for the ML approach
#  such as not normalising the descriptors.
# creates 2d-3d matches data for ransac comparison
import os
import time
from itertools import chain
import cv2
import numpy as np
import sys
import subprocess
from os.path import exists

# creates 2d-3d matches data for ransac comparison
from show_2D_points_predicting_matchability import show_projected_points


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

def feature_matcher_wrapper_predicting_matchability(base_path, comparison_data_path, db, query_images, trainDescriptors, points3D_xyz, ratio_test_val, top_no = None, verbose= True):
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    matches_sum = []
    total_time = 0
    percentage_reduction_total = 0
    image_gt_dir = os.path.join(base_path, 'gt/images/')
    vlfeat_command_path = "code_to_compare/Predicting_Matchability/VLFeat_SIFT/VLFeat_SIFT"
    predicting_matchability_random_forest = "code_to_compare/Predicting_Matchability/VLFeat_SIFT/rforest.gz"

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in range(len(query_images)):
        query_image = query_images[i]
        if(verbose):
            print("Matching image " + str(i + 1) + "/" + str(len(query_images)) + ", " + query_image)

        image_gt_path = os.path.join(image_gt_dir, query_image)

        if( exists(comparison_data_path) == False):
            print("comparison_data_path does not exist")
            exit()

        query_image_no_folder = query_image.split("/")[1]
        converted_image_gt_path = os.path.join(comparison_data_path, query_image_no_folder.replace(".jpg", ".pgm")) #split here is to get rid of the "session7/" folder name

        if(exists(converted_image_gt_path) == False):
            # convert image for VLFeat (required imagemagick)
            convert_command = ["convert", image_gt_path, converted_image_gt_path]
            subprocess.check_call(convert_command)

        converted_image_gt_sift_path_all = converted_image_gt_path.replace(".pgm", ".sift_all")
        converted_image_gt_sift_path_classify = converted_image_gt_path.replace(".pgm", ".sift_classified")
        vlfeat_command_no_classify = [vlfeat_command_path, "--octaves", "2", "--levels", "3", "--first-octave", "0", "--peak-thresh", "0.001", "--edge-thresh", "10.0", "--magnif", "3", "--output", converted_image_gt_sift_path_all, converted_image_gt_path]
        vlfeat_command_classify = [vlfeat_command_path, "--octaves", "2", "--levels", "3", "--first-octave", "0", "--peak-thresh", "0.001", "--edge-thresh", "10.0", "--magnif", "3", "--classify", predicting_matchability_random_forest, "--cl-thresh", "0.525", "--output", converted_image_gt_sift_path_classify, converted_image_gt_path]

        # original
        subprocess.check_call(vlfeat_command_no_classify)
        #  just to get the length
        keypoints_xy_descs = np.loadtxt(converted_image_gt_sift_path_all)
        len_descs_all_classify = keypoints_xy_descs.shape[0]

        # predicting matchability code
        #  overwrites the "converted_image_gt_sift_path" file
        start = time.time()
        subprocess.check_call(vlfeat_command_classify) #calling predicting matchability code
        end = time.time()
        elapsed_time = end - start
        total_time += elapsed_time

        # This is to generate visuals for my thesis
        print("Running show_projected_points() for " + image_gt_path)
        show_projected_points(image_gt_path, comparison_data_path, query_image_no_folder, converted_image_gt_sift_path_all, converted_image_gt_sift_path_classify)

        keypoints_xy_descs = np.loadtxt(converted_image_gt_sift_path_classify)
        keypoints_xy = keypoints_xy_descs[:,0:2]
        queryDescriptors = keypoints_xy_descs[:,4:132]
        len_descs = queryDescriptors.shape[0]

        percentage_reduction_total = percentage_reduction_total + (100 - len_descs * 100 / len_descs_all_classify)

        queryDescriptors = queryDescriptors.astype(np.float32)  # required for opencv
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

        matches[query_image] = np.array(good_matches)
        matches_sum.append(len(good_matches))

    if(verbose):
        print()
        total_all_images = np.sum(matches_sum)
        print("Total matches: " + str(total_all_images) + ", no of images " + str(len(query_images)))
        matches_all_avg = total_all_images / len(matches_sum)
        print("Average matches per image: " + str(matches_all_avg) + ", no of images " + str(len(query_images)))
        percentage_reduction_avg = percentage_reduction_total / len(query_images)
        print("Average matches percentage reduction per image: " + str(percentage_reduction_avg) + "%")

    total_avg_time = total_time / len(query_images)
    return matches, total_avg_time