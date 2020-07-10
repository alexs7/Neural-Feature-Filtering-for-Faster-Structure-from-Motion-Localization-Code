# This is to match sfm images (already localised in COLMAP) descs against the
# trainDescriptors (which can from the base model or the complete model) as a benchmark and also exports the 2D-3D matches for ransac
# matching here is done using my own DM (direct matching) function.

# NOTE: One can argue why am I using the query images only (query_name.txt)? It makes sense more intuitively as
# I am localising the new (future sessions images) against a base model and a complete model. So the difference is in
# the model you are localising against.. But you could use all images. If you do then localising base images against the
# base model doesn't really makes sense, because at this point you are localising images the model has already seen but then again
# you can say the same thing for localising future images against the complete model
import cv2
import numpy as np
from point3D_loader import read_points3d_default, index_dict

# creates 2d-3d matches data for ransac
def get_matches(good_matches_data, points3D_indexing, points3D, query_image_xy, scores):
    # same length
    # good_matches_data[0] - 2D point indices,
    # good_matches_data[1] - 3D point indices, - this is the index you need the id to get xyz
    # good_matches_data[2] - lowe's distance inverse ratio
    # good_matches_data[3] - reliability scores ratio
    # good_matches_data[4] - reliability_scores = lowe's distance inverse * reliability scores ratio
    data_size = 10
    matches = np.empty([0, data_size])
    for i in range(len(good_matches_data[1])):
        # get 2D point data
        xy_2D = query_image_xy[good_matches_data[0][i]]

        # get 3D point data
        points3D_index = good_matches_data[1][i] # remember points3D_index is aligned with trainDescriptors
        points3D_id = points3D_indexing[points3D_index]
        xyz_3D = points3D[points3D_id].xyz

        # get lowe's inv ratio
        lowes_distance_inverse_ratio = good_matches_data[2][i]
        # reliability scores ratio
        reliability_score_ratio = good_matches_data[3][i]
        # reliability_scores
        reliability_score = good_matches_data[4][i]

        # the heatmap score value (of the 3D point of the match, the closest match, m)
        heat_map_val = scores[0, points3D_index]

        # values here are self explanatory..
        match = np.array([xy_2D[0], xy_2D[1], xyz_3D[0], xyz_3D[1], xyz_3D[2], points3D_index,
                          lowes_distance_inverse_ratio, heat_map_val, reliability_score_ratio, reliability_score]).reshape([1, data_size])
        matches = np.r_[matches, match]
    return matches

# indexing is the same as points3D indexing for trainDescriptors
def feature_matcher_wrapper(scores, db, query_images, trainDescriptors, points3D, ratio_test_val, verbose = False):
    # create image_name <-> matches, dict - easier to work with
    matches = {}
    matches_sum = []
    points3D_indexing = index_dict(points3D)

    #  go through all the test images and match their descs to the 3d points avg descs
    for i in range(len(query_images)):
        query_image = query_images[i]
        if(verbose): print("Matching image " + str(i + 1) + "/" + str(len(query_images)) + ", " + query_image, end="\r")

        image_id = db.execute("SELECT image_id FROM images WHERE name = " + "'" + query_image + "'")
        image_id = str(image_id.fetchone()[0])

        # fetching the (x,y,descs) for that image
        query_image_descriptors_data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + image_id + "'")
        query_image_descriptors_data = query_image_descriptors_data.fetchone()[0]
        query_image_descriptors_data = db.blob_to_array(query_image_descriptors_data, np.uint8)
        descs_rows = int(np.shape(query_image_descriptors_data)[0] / 128)
        query_image_descriptors_data = query_image_descriptors_data.reshape([descs_rows, 128])

        # once you have the test images descs now do feature matching here! - Matching on trainDescriptors (remember these are the means of the 3D points)
        queryDescriptors = query_image_descriptors_data.astype(np.float32)

        # actual matching here!
        # NOTE: 09/06/2020 - match() has been changed to return lowes_distances in REVERSE! (https://willguimont.github.io/cs/2019/12/26/prosac-algorithm.html)
        # good_matches = matcher.match(queryDescriptors, trainDescriptors)

        # NOTE: 03/07/2020 - using matching method from OPENCV
        bf = cv2.BFMatcher()
        temp_matches = bf.knnMatch(queryDescriptors, trainDescriptors, k=2)

        # output: idx1, idx2, lowes_distance (vectors of corresponding indexes in
        # queryDescriptors and trainDescriptors, and lowes_distances inverse respectively)
        idx1, idx2, lowes_distances, reliability_ratios, reliability_scores  = [], [], [], [], []
        # m the closest, n is the second closest
        for m, n in temp_matches:
            if m.distance < matcher.ratio_test * n.distance: #TODO: change to also use reliability score ?
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                score_m = scores[0, m.trainIdx]
                score_n = scores[0, n.trainIdx]
                lowes_distance_inverse = n.distance / m.distance #inverse here as the higher the better for PROSAC
                lowes_distances.append(lowes_distance_inverse)
                reliability_score_ratio = score_m / score_n # the higher the better (first match is more "static" than the second, ratio)
                reliability_ratios.append(reliability_score_ratio)
                reliability_score = lowes_distance_inverse * reliability_score_ratio # self-explanatory
                reliability_scores.append(reliability_score)
        # at this point you store 1, 2D - 3D match.
        good_matches = [idx1, idx2, lowes_distances, reliability_ratios, reliability_scores]
        # queryDescriptors and query_image_keypoints_data_xy = same order
        # points3D order and trainDescriptors_* = same order
        # returns extra data for each match
        matches[query_image] = get_matches(good_matches, points3D_indexing, points3D, query_image_keypoints_data_xy, scores)
        matches_sum.append(len(good_matches[0]))

    if(verbose == True):
        print()
        total_all_images = np.sum(matches_sum)
        print("Total matches: " + str(total_all_images) + ", no of images " + str(len(query_images)))
        matches_all_avg = total_all_images / len(matches_sum)
        print("Average matches per image: " + str(matches_all_avg) + ", no of images " + str(len(query_images)))

    return matches

