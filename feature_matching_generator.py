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
from database import COLMAPDatabase, blob_to_array

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

def feature_matcher_wrapper(points3D_avg_heatmap_vals, db_path, images_path, train_descriptors_path, points3D, points3D_indexing, matcher):

    db = COLMAPDatabase.connect(db_path)

    # images to use for testing
    # NOTE: Here you can use the localised images that contain also the base images but there is no point ..
    test_images = load_images_from_text_file(images_path)

    # this loads the descs means for the base model and the complete model indexing is the same as points3D indexing
    trainDescriptors = np.load(train_descriptors_path)
    trainDescriptors = trainDescriptors.astype(np.float32)

    # create image_name <-> value dict - easier to work with
    matches = {}
    matches_sum = []

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in range(len(test_images)):
        test_image = test_images[i]
        print("Matching image " + str(i + 1) + "/" + str(len(test_images)) + ", " + test_image, end="\r")

        image_id = db.execute("SELECT image_id FROM images WHERE name = " + "'" + test_image + "'")
        image_id = str(image_id.fetchone()[0])
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
        # NOTE: 09/06/2020 - match() has been changed to return lowes_distances in REVERSE! (https://willguimont.github.io/cs/2019/12/26/prosac-algorithm.html)
        good_matches = matcher.match(queryDescriptors, trainDescriptors)
        # good_matches_base = matcher.match(queryDescriptors, trainDescriptors_base)

        # queryDescriptors and query_image_keypoints_data_xy = same order
        # points3D order and trainDescriptors_* = same order
        # returns extra data for each match
        matches[test_image] = get_matches(good_matches, points3D_indexing, points3D, query_image_keypoints_data_xy, points3D_avg_heatmap_vals)
        matches_sum.append(len(good_matches[0]))

    print()
    matches_all_avg = np.sum(matches_sum) / len(matches_sum)
    print("Average matches per image: " + str(matches_all_avg) + ", no of images " + str(len(test_images)) )

    return matches

