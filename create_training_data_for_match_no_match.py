# Run this after format_data_for_match_no_match.py
import glob
import os
import sys
import cv2
import pycolmap
import shutil
import colmap
from database import COLMAPDatabase
from database import pair_id_to_image_ids
import numpy as np
import random
from tqdm import tqdm
from parameters import Parameters
from point3D_loader import read_points3d_default
from query_image import is_image_from_base, read_images_binary, get_image_name_from_db_with_id, get_all_images_names_from_db, get_image_id, \
    get_keypoints_data_and_dominantOrientations, get_descriptors


def get_image_decs(db, image_id): #not to be confused with get_queryDescriptors() in feature_matching_generator.py - that one normalises descriptors.
    data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(image_id) + "'")
    data = COLMAPDatabase.blob_to_array(data.fetchone()[0], np.uint8)
    descs_rows = int(np.shape(data)[0] / 128)
    descs = data.reshape([descs_rows, 128])  # descs for the whole image
    return descs

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

def get_image_keypoints_data(db, img_id):
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

def get_full_path(lpath, bpath, name):
    if (('session' in name) == True):
        image_path = os.path.join(lpath, name)
    else:
        image_path = os.path.join(bpath, name)  # then it is a base model image (still call it live for convinience)
    return image_path

def get_subset_of_pairs(all_pair_ids, no):
    random.shuffle(all_pair_ids)
    img_left_ids = []
    img_right_ids = []
    pair_ids = []
    pbar = tqdm(total=no)
    while (len(pair_ids) < no):
        rnd_pair_id = random.choice(all_pair_ids)
        img_id_1, img_id_2 = pair_id_to_image_ids(rnd_pair_id[0])
        if ((img_id_1 not in img_left_ids) and (img_id_2 not in img_right_ids)):
            pair_ids.append(rnd_pair_id)  # which is a tuple
            pbar.update(1)
    pbar.close()
    return pair_ids

def createDataForMatchNoMatchMatchabilityComparison(image_base_dir, image_live_dir, db_live, live_images, output_db_path, pairs_limit=-1):
    print("Getting Pairs")
    if(pairs_limit == -1):
        pair_ids = db_live.execute("SELECT pair_id FROM matches").fetchall()
    else:
        all_pair_ids = db_live.execute("SELECT pair_id FROM matches").fetchall()
        pair_ids = get_subset_of_pairs(all_pair_ids, pairs_limit) #as in paper

    print("Creating data..")
    training_data_db = COLMAPDatabase.create_db_match_no_match_data(os.path.join(output_db_path, "training_data_small.db"))
    training_data_db.execute("BEGIN")

    for pair in tqdm(pair_ids):
        pair_id = pair[0]
        img_id_1, img_id_2 = pair_id_to_image_ids(pair_id)

        # when a db image has not been localised ...
        if((img_id_1 not in live_images) or (img_id_2 not in live_images)):
            continue

        img_1_file_name = live_images[img_id_1].name
        img_2_file_name = live_images[img_id_2].name

        img_1_file = cv2.imread(get_full_path(image_live_dir, image_base_dir, img_1_file_name))
        img_2_file = cv2.imread(get_full_path(image_live_dir, image_base_dir, img_2_file_name))

        descs_img1 = get_image_decs(db_live, img_id_1)
        descs_img2 = get_image_decs(db_live, img_id_2)

        pair_data = db_live.execute("SELECT rows, data FROM matches WHERE pair_id = " + "'" + str(pair_id) + "'").fetchone()
        rows = pair_data[0]

        if(rows < 1): #no matches in this pair, no idea why COLMAP stores it...
            continue

        cols = 2 #for each image
        zero_based_indices = COLMAPDatabase.blob_to_array(pair_data[1], np.uint32).reshape([rows, cols])
        zero_based_indices_left = zero_based_indices[:, 0]
        zero_based_indices_right = zero_based_indices[:, 1]

        keypoints_data_img_1 = get_image_keypoints_data(db_live, img_id_1)
        keypoints_data_img_2 = get_image_keypoints_data(db_live, img_id_2)

        keypoints_data_img_1_matched = keypoints_data_img_1[zero_based_indices_left]
        keypoints_data_img_1_matched_descs = descs_img1[zero_based_indices_left]
        keypoints_data_img_1_unmatched = np.delete(keypoints_data_img_1, zero_based_indices_left, axis=0)
        keypoints_data_img_1_unmatched_descs = np.delete(descs_img1, zero_based_indices_left, axis=0)

        keypoints_data_img_2_matched = keypoints_data_img_2[zero_based_indices_right]
        keypoints_data_img_2_matched_descs = descs_img2[zero_based_indices_right]
        keypoints_data_img_2_unmatched = np.delete(keypoints_data_img_2, zero_based_indices_right, axis=0)
        keypoints_data_img_2_unmatched_descs = np.delete(descs_img2, zero_based_indices_right, axis=0)

        # matched, img_1
        for i in range(keypoints_data_img_1_matched.shape[0]):
            sample = keypoints_data_img_1_matched[i,:]
            xy = sample[0:2]
            desc = keypoints_data_img_1_matched_descs[i,:]
            desc_scale = sample[2]
            desc_orientation = sample[3]
            live_image_file_xy_green_intensity = img_1_file[int(xy[1]), int(xy[0])][1]  # reverse indexing
            training_data_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                          (img_id_1,) + (COLMAPDatabase.array_to_blob(desc),) + (1,) + (desc_scale,) +
                          (desc_orientation,) + (xy[0],) + (xy[1],) + (int(live_image_file_xy_green_intensity),))
        # unmatched, img_1
        for i in range(keypoints_data_img_1_unmatched.shape[0]):
            sample = keypoints_data_img_1_unmatched[i, :]
            xy = sample[0:2]
            desc = keypoints_data_img_1_unmatched_descs[i, :]
            desc_scale = sample[2]
            desc_orientation = sample[3]
            live_image_file_xy_green_intensity = img_1_file[int(xy[1]), int(xy[0])][1]  # reverse indexing
            training_data_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                     (img_id_1,) + (COLMAPDatabase.array_to_blob(desc),) + (0,) + (desc_scale,) + (desc_orientation,) + (xy[0],) + (xy[1],) + (
                                     int(live_image_file_xy_green_intensity),))
        # matched, img_2
        for i in range(keypoints_data_img_2_matched.shape[0]):
            sample = keypoints_data_img_2_matched[i, :]
            xy = sample[0:2]
            desc = keypoints_data_img_2_matched_descs[i, :]
            desc_scale = sample[2]
            desc_orientation = sample[3]
            live_image_file_xy_green_intensity = img_2_file[int(xy[1]), int(xy[0])][1]  # reverse indexing
            training_data_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                     (img_id_2,) + (COLMAPDatabase.array_to_blob(desc),) + (1,) + (desc_scale,) + (desc_orientation,) + (xy[0],) + (xy[1],) + (
                                     int(live_image_file_xy_green_intensity),))
        # unmatched, img_2
        for i in range(keypoints_data_img_2_unmatched.shape[0]):
            sample = keypoints_data_img_2_unmatched[i, :]
            xy = sample[0:2]
            desc = keypoints_data_img_2_unmatched_descs[i, :]
            desc_scale = sample[2]
            desc_orientation = sample[3]
            live_image_file_xy_green_intensity = img_2_file[int(xy[1]), int(xy[0])][1]  # reverse indexing
            training_data_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                     (img_id_2,) + (COLMAPDatabase.array_to_blob(desc),) + (0,) + (desc_scale,) + (desc_orientation,) + (xy[0],) + (xy[1],) + (
                                     int(live_image_file_xy_green_intensity),))

    print("Committing..")
    training_data_db.commit()
    print("Generating Stats..")
    stats = training_data_db.execute("SELECT * FROM data").fetchall()
    matched = training_data_db.execute("SELECT * FROM data WHERE matched = 1").fetchall()
    unmatched = training_data_db.execute("SELECT * FROM data WHERE matched = 0").fetchall()

    print("Total samples: " + str(len(stats)))
    print("Total matched samples: " + str(len(matched)))
    print("Total unmatched samples: " + str(len(unmatched)))
    print("% of matched samples: " + str(len(matched) * 100 / len(unmatched)))

    print("Done!")

base_path = sys.argv[1]
pairs_limit = int(sys.argv[2])
print("Base path: " + base_path)
parameters = Parameters(base_path)
db_live = COLMAPDatabase.connect(parameters.live_db_path)
live_model_images = read_images_binary(parameters.live_model_images_path)
live_model_points3D = read_points3d_default(parameters.live_model_points3D_path)
image_live_dir = os.path.join(base_path, 'live/images/')
image_base_dir = os.path.join(base_path, 'base/images/')

db_live_path = os.path.join(base_path, "live/database.db")
output_path = os.path.join(base_path, "match_or_no_match_comparison_data")
os.makedirs(output_path, exist_ok = True)
createDataForMatchNoMatchMatchabilityComparison(image_base_dir, image_live_dir, db_live, live_model_images, output_path, pairs_limit = pairs_limit)
