# This is to match sfm images (already localised) descs against the
# base model and the complete model as a benchmark and also exports the 2D-3D matches for ransac
# matching here is done using my own DM (direct matching) function.

# NOTE: One can argue why am I using the query images only (query_name.txt)? It makes sense more intuitively as
# I am localising the new (future sessions images) against a base model and a complete model. So the difference is in
# the model you are localising against.. But you could use all images. If you do then localising base images against the
# base model doesn't really makes sense, because at this point you are localising images the model has already seen but then again
# you can say the same thing for localising future images against the complete model

import sqlite3

import numpy as np
import sys
import cv2

from feature_matcher import feature_matcher_factory, FeatureMatcherTypes
from parameters import Parameters
from point3D_loader import read_points3d_default
from query_image import read_images_binary, image_localised, load_images_from_text_file

IS_PYTHON3 = sys.version_info[0] >= 3

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

# creates 2d-3d matches data for ransac
def get_matches(good_matches_data, points3D_indexing, points3D, query_image_xy, points3D_avg_heatmap_vals):
    # same length
    # good_matches_data[0] - 2D point indices,
    # good_matches_data[1] - 3D point indices, - this is the index you need the id to get xyz
    # good_matches_data[2] lowe's distance inverse ratio
    data_size = 8
    matches = np.empty([0, data_size])
    for i in range(len(good_matches_data[1])):
        # get 3D point data
        points3D_index = good_matches_data[1][i]
        points3D_id = points3D_indexing[points3D_index]
        xyz_3D = points3D[points3D_id].xyz
        # get 2D point data
        xy_2D = query_image_xy[good_matches_data[0][i]]
        # remember points3D_index is aligned with trainDescriptors_*
        lowes_distance_inverse_ratio = good_matches_data[2][i]
        # the heatmap dist value (points3D_avg_heatmap_vals.sum() = 1)
        heat_map_val = points3D_avg_heatmap_vals[0,points3D_index]
        # values here are self explanatory..
        match = np.array([xy_2D[0], xy_2D[1], xyz_3D[0], xyz_3D[1], xyz_3D[2], points3D_index, lowes_distance_inverse_ratio, heat_map_val]).reshape([1, data_size])
        matches = np.r_[matches, match]
    return matches

def feature_matcher_wrapper(features_no, points3D_avg_heatmap_vals):

    print("-- Matching features_no " + features_no + " --")

    db = COLMAPDatabase.connect("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/database.db")

    # by "complete model" I mean all the frames from future sessions localised in the base model (28/03)
    complete_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/images.bin"
    complete_model_all_images = read_images_binary(complete_model_images_path)
    complete_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/points3D.bin"
    points3D = read_points3d_default(complete_model_points3D_path) # base model's 3D points (same length as complete as we do not add points when localising new points, but different image_ds for each point)

    # create points id and index relationship
    point3D_index = 0
    points3D_indexing = {}
    for key, value in points3D.items():
        points3D_indexing[point3D_index] = value.id
        point3D_index = point3D_index + 1

    # images to use for testing
    # NOTE: Here you can use the localised images that contain also the base images but there is no point ..
    test_images = load_images_from_text_file("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_name.txt")

    print("Loading 3D Points mean descs")
    # this loads the descs means for the base model and the complete model indexing is the same as points3D indexing
    trainDescriptors_all = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/"+features_no+"/points_mean_descs_all.npy")
    trainDescriptors_base = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/"+features_no+"/points_mean_descs_base.npy")
    trainDescriptors_all = trainDescriptors_all.astype(np.float32)
    trainDescriptors_base = trainDescriptors_base.astype(np.float32)

    # create image_name <-> value dict - easier to work with
    results_base = {}
    results_all = {}

    matches_base = {}
    matches_all = {}

    matches_base_sum = []
    matches_all_sum = []

    #  go through all the test images and match their descs to the 3d points avg descs
    for test_image in test_images:
        image_id = image_localised(test_image, complete_model_all_images)

        if(image_id != None):
            image_id = str(image_id)

            # fetching the x,y,descs for that image
            query_image_keypoints_data = db.execute("SELECT data FROM keypoints WHERE image_id = " + "'" + image_id + "'")
            query_image_keypoints_data = query_image_keypoints_data.fetchone()[0]
            query_image_keypoints_data_cols = db.execute("SELECT cols FROM keypoints WHERE image_id = " + "'" + image_id + "'")
            query_image_keypoints_data_cols = int(query_image_keypoints_data_cols.fetchone()[0])
            query_image_keypoints_data = blob_to_array(query_image_keypoints_data, np.float32)
            query_image_keypoints_data_rows = int(np.shape(query_image_keypoints_data)[0] / query_image_keypoints_data_cols)
            query_image_keypoints_data = query_image_keypoints_data.reshape(query_image_keypoints_data_rows,query_image_keypoints_data_cols)
            query_image_keypoints_data_xy = query_image_keypoints_data[:, 0:2]
            query_image_descriptors_data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + image_id + "'")
            query_image_descriptors_data = query_image_descriptors_data.fetchone()[0]
            query_image_descriptors_data = blob_to_array(query_image_descriptors_data, np.uint8)
            descs_rows = int(np.shape(query_image_descriptors_data)[0] / 128)
            query_image_descriptors_data = query_image_descriptors_data.reshape([descs_rows, 128])

            # once you have the test images descs now do feature matching here! - Matching on all and base descs means
            queryDescriptors = query_image_descriptors_data.astype(np.float32)

            # actual matching here!
            matching_algo = FeatureMatcherTypes.FLANN  # or FeatureMatcherTypes.BF
            match_ratio_test = Parameters.kFeatureMatchRatioTest
            norm_type = cv2.NORM_L2
            cross_check = False
            matcher = feature_matcher_factory(norm_type, cross_check, match_ratio_test, matching_algo)
            # NOTE: 09/06/2020 - match() has been changed to return lowes_distances in REVERSE! (https://willguimont.github.io/cs/2019/12/26/prosac-algorithm.html)
            good_matches_all = matcher.match(queryDescriptors, trainDescriptors_all)
            good_matches_base = matcher.match(queryDescriptors, trainDescriptors_base)

            # remember these are only indices
            results_all[test_image] = len(good_matches_all[0])
            results_base[test_image] = len(good_matches_base[0])

            # queryDescriptors and query_image_keypoints_data_xy = same order
            # points3D order and trainDescriptors_* = same order
            matches_all[test_image] = get_matches(good_matches_all, points3D_indexing, points3D, query_image_keypoints_data_xy, points3D_avg_heatmap_vals)
            matches_base[test_image] = get_matches(good_matches_base, points3D_indexing, points3D, query_image_keypoints_data_xy, points3D_avg_heatmap_vals)

            matches_base_sum.append(len(good_matches_base[0]))
            matches_all_sum.append(len(good_matches_all[0]))

    print("Averages: ")
    matches_base_avg = np.sum(matches_base_sum) / len(matches_base_sum)
    print("Base model avg matches: " + str(matches_base_avg))
    matches_all_avg = np.sum(matches_all_sum) / len(matches_all_sum)
    print("Complete model avg matches: " + str(matches_all_avg))

    print("Saving data...")
    # save the 2D-3D matches
    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/"+features_no+"/matches_all.npy", matches_all)
    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/"+features_no+"/matches_base.npy", matches_base)

    # again this is mostly of visualing results
    # results_* contain the numbers of matches for each image so, length will be the same as the localised images no.
    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/"+features_no+"/results_all.npy", results_all)
    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/"+features_no+"/results_base.npy", results_base)
    print("Done!")

def feature_matcher_wrapper_weighted(features_no, exponential_decay_value, points3D_avg_heatmap_vals):

    print("-- Matching features_no " + features_no + " --")

    db = COLMAPDatabase.connect("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/database.db")

    # by "complete model" I mean all the frames from future sessions localised in the base model (28/03)
    complete_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/images.bin"
    complete_model_all_images = read_images_binary(complete_model_images_path)
    complete_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/points3D.bin"
    points3D = read_points3d_default(complete_model_points3D_path) # base model's 3D points (same length as complete as we do not add points when localising new points, but different image_ds for each point)

    # create points id and index relationship
    point3D_index = 0
    points3D_indexing = {}
    for key, value in points3D.items():
        points3D_indexing[point3D_index] = value.id
        point3D_index = point3D_index + 1

    # images to use for testing
    # NOTE: Here you can use the localised images that contain also the base images but there is no point ..
    test_images = load_images_from_text_file("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_name.txt")

    print("Loading 3D Points weighted descs")
    # this loads the descs means for the base model and the complete model indexing is the same as points3D indexing
    trainDescriptors_all = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/"+features_no+"/points_weighted_descs_all_with_extra_data_"+str(exponential_decay_value)+".npy")
    trainDescriptors_base = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/"+features_no+"/points_mean_descs_base.npy")
    trainDescriptors_all = trainDescriptors_all.astype(np.float32)
    trainDescriptors_base = trainDescriptors_base.astype(np.float32)

    # create image_name <-> value dict - easier to work with
    results_base = {}
    results_all = {}

    matches_base = {}
    matches_all = {}

    matches_base_sum = []
    matches_all_sum = []

    #  go through all the test images and match their descs to the 3d points avg descs
    for test_image in test_images:
        image_id = image_localised(test_image, complete_model_all_images)

        if(image_id != None):
            image_id = str(image_id)

            # fetching the x,y,descs for that image
            query_image_keypoints_data = db.execute("SELECT data FROM keypoints WHERE image_id = " + "'" + image_id + "'")
            query_image_keypoints_data = query_image_keypoints_data.fetchone()[0]
            query_image_keypoints_data_cols = db.execute("SELECT cols FROM keypoints WHERE image_id = " + "'" + image_id + "'")
            query_image_keypoints_data_cols = int(query_image_keypoints_data_cols.fetchone()[0])
            query_image_keypoints_data = blob_to_array(query_image_keypoints_data, np.float32)
            query_image_keypoints_data_rows = int(np.shape(query_image_keypoints_data)[0] / query_image_keypoints_data_cols)
            query_image_keypoints_data = query_image_keypoints_data.reshape(query_image_keypoints_data_rows,query_image_keypoints_data_cols)
            query_image_keypoints_data_xy = query_image_keypoints_data[:, 0:2]
            query_image_descriptors_data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + image_id + "'")
            query_image_descriptors_data = query_image_descriptors_data.fetchone()[0]
            query_image_descriptors_data = blob_to_array(query_image_descriptors_data, np.uint8)
            descs_rows = int(np.shape(query_image_descriptors_data)[0] / 128)
            query_image_descriptors_data = query_image_descriptors_data.reshape([descs_rows, 128])

            # once you have the test images descs now do feature matching here! - Matching on all and base descs means
            queryDescriptors = query_image_descriptors_data.astype(np.float32)

            # actual matching here!
            matching_algo = FeatureMatcherTypes.FLANN  # or FeatureMatcherTypes.BF
            match_ratio_test = Parameters.kFeatureMatchRatioTest
            norm_type = cv2.NORM_L2
            cross_check = False
            matcher = feature_matcher_factory(norm_type, cross_check, match_ratio_test, matching_algo)
            good_matches_all = matcher.match(queryDescriptors, trainDescriptors_all)
            good_matches_base = matcher.match(queryDescriptors, trainDescriptors_base)

            # remember these are only indices
            results_all[test_image] = len(good_matches_all[0])
            results_base[test_image] = len(good_matches_base[0])

            # queryDescriptors and query_image_keypoints_data_xy = same order
            # points3D order and trainDescriptors_* = same order
            matches_all[test_image] = get_matches(good_matches_all, points3D_indexing, points3D, query_image_keypoints_data_xy, points3D_avg_heatmap_vals)
            matches_base[test_image] = get_matches(good_matches_base, points3D_indexing, points3D, query_image_keypoints_data_xy, points3D_avg_heatmap_vals)

            matches_base_sum.append(len(good_matches_base[0]))
            matches_all_sum.append(len(good_matches_all[0]))

    print("Averages: ")
    matches_base_avg = np.sum(matches_base_sum) / len(matches_base_sum)
    print("Base model avg matches: " + str(matches_base_avg))
    matches_all_avg = np.sum(matches_all_sum) / len(matches_all_sum)
    print("Complete model avg matches: " + str(matches_all_avg))

    print("Saving data...")
    # save the 2D-3D matches
    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/"+features_no+"/matches_all_weighted.npy", matches_all)
    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/"+features_no+"/matches_base_weighted.npy", matches_base)

    # again this is mostly of visualing results
    # results_* contain the numbers of matches for each image so, length will be the same as the localised images no.
    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/"+features_no+"/results_all.npy", results_all)
    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/"+features_no+"/results_base.npy", results_base)
    print("Done!")

# colmap_features_no can be "2k", "1k", "0.5k", "0.25k"
# exponential_decay can be any of 0.1 to 0.9
features_no = "1k"
exponential_decay_value = 0.5

#distribution; row vector, same size as 3D points
points3D_avg_heatmap_vals = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/"+features_no+"/heatmap_matrix_avg_points_values_" + str(exponential_decay_value) + ".txt")
points3D_avg_heatmap_vals = points3D_avg_heatmap_vals.reshape([1, points3D_avg_heatmap_vals.shape[0]])

print("Doing Vanilla feature matching")
feature_matcher_wrapper(features_no, points3D_avg_heatmap_vals)

print("Doing weighted feature matching")
feature_matcher_wrapper_weighted(features_no, exponential_decay_value, points3D_avg_heatmap_vals)
