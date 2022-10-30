# This file will contain methods to generate the gt data from all query (gt images) to test with certain models
# such as PM and MnM
# TODO: Move the testing data generation from create_training_data_and_train_for_match_no_match.py to here

import os
import cv2
import numpy as np
from tqdm import tqdm
from database import COLMAPDatabase
from point3D_loader import read_points3d_default
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_image_id, get_descriptors, get_image_data

def createGTTestDataForPM(base_path):
    # get test data too (gt = query as we know)
    db_gt_path = os.path.join(base_path, "gt/database.db")
    db_gt = COLMAPDatabase.connect(db_gt_path)

    query_images_bin_path = os.path.join(base_path, "gt/model/images.bin")
    query_images_names = load_images_from_text_file(os.path.join(base_path, "gt/query_name.txt"))
    localised_query_images_names = get_localised_image_by_names(query_images_names, query_images_bin_path)

    gt_points_3D = read_points3d_default(os.path.join(base_path, "gt/model/points3D.bin"))
    gt_model_images = read_images_binary(query_images_bin_path)

    data_to_write = np.empty([0,129])
    for loc_img_name in tqdm(localised_query_images_names):
        image_id = get_image_id(db_gt, loc_img_name)
        image = gt_model_images[int(image_id)]
        kp_db_row = db_gt.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = " + "'" + str(image_id) + "'").fetchone()
        rows = kp_db_row[0]
        _, _, descs = get_descriptors(db_gt, image_id)
        assert (image.xys.shape[0] == image.point3D_ids.shape[0] == rows == descs.shape[0])  # just for my sanity
        matched_values = []  # for each keypoint (x,y)/desc same thing

        for i in range(image.xys.shape[0]):  # can loop through descs or img_data.xys - same order
            current_point3D_id = image.point3D_ids[i]
            if (current_point3D_id == -1):  # means feature is unmatched
                matched = 0
            else:
                # this is to make sure that xy belong to the right pointd3D
                assert i in gt_points_3D[current_point3D_id].point2D_idxs
                matched = 1
            matched_values.append(matched)

        matched_values = np.array(matched_values).reshape(rows, 1)
        data = np.c_[descs, matched_values]
        data_to_write = np.r_[data_to_write, data]
    return data_to_write.astype(np.float32)

# Note that this will return less data than the PM equivalent as less query_name.txt are localised
def createGTTestDataForMnM(base_path, mnm_base_path):
    db_gt_mnm_path = os.path.join(mnm_base_path, "gt/database.db") #openCV db + extra MnM data
    db_gt_mnm = COLMAPDatabase.connect(db_gt_mnm_path)  # remember this database holds the OpenCV descriptors
    query_images_bin_path_mnm = os.path.join(mnm_base_path, "gt/output_opencv_sift_model/images.bin")

    query_images_path = os.path.join(base_path, "gt/query_name.txt")  # these are the same anw
    query_images_names = load_images_from_text_file(query_images_path)
    localised_query_images_names_mnm = get_localised_image_by_names(query_images_names, query_images_bin_path_mnm)

    image_gt_dir_mnm = os.path.join(base_path, 'gt/images/') # or mnm_base_path should be the same
    gt_points_3D_mnm = read_points3d_default(os.path.join(mnm_base_path, "gt/output_opencv_sift_model/points3D.bin"))
    gt_model_images_mnm = read_images_binary(query_images_bin_path_mnm)

    data_to_write = [] #size = localised_query_images_names_mnm * test_data_img (for each image)
    for i in tqdm(range(len(localised_query_images_names_mnm))):
        img_name = localised_query_images_names_mnm[i]
        image_gt_path = os.path.join(image_gt_dir_mnm, img_name)
        qt_image_file = cv2.imread(image_gt_path)  # no need for cv2.COLOR_BGR2RGB here as G is in the middle anw
        image_id = int(get_image_id(db_gt_mnm, img_name))
        test_data_img = get_image_data(db_gt_mnm, gt_points_3D_mnm, gt_model_images_mnm, image_id, qt_image_file)

        for i in range(len(test_data_img)): #as many keypoints as detected
            sample = test_data_img[i, :]
            x = sample[0]
            y = sample[1]
            octave = sample[2]
            angle = sample[3]
            size = sample[4]
            response = sample[5]
            green_intensity = sample[6]
            dominantOrientation = sample[7]
            matched = sample[8] #can use astype(np.int64) here
            data = np.array([x,y,octave,angle,size,response,green_intensity,dominantOrientation,matched])
            data_to_write.append(data.reshape([1, len(data)]))

    return np.array(data_to_write).reshape([len(data_to_write), len(data)]).astype(np.float32)
