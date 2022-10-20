# Run this after format_data_for_match_no_match.py
import glob
import os
import sys
import cv2
from database import COLMAPDatabase
from database import pair_id_to_image_ids
import numpy as np
import random
from tqdm import tqdm
from query_image import read_images_binary

def get_image_keypoints_data(db, img_id):
    kp_db_row = db.execute("SELECT rows, cols, data, dominantOrientations FROM keypoints WHERE image_id = " + "'" + str(img_id) + "'").fetchone()
    cols = kp_db_row[1]
    rows = kp_db_row[0]
    # x, y, octave, angle, size, response
    kp_data = COLMAPDatabase.blob_to_array(kp_db_row[2], np.float32)
    kp_data = kp_data.reshape([rows, cols])
    dominantOrientations = COLMAPDatabase.blob_to_array(kp_db_row[3], np.uint8)
    dominantOrientations = dominantOrientations.reshape([rows, 1])
    return np.c_[kp_data, dominantOrientations]

def get_full_path(lpath, bpath, name):
    if (('session' in name) == True):
        image_path = os.path.join(lpath, name)
    else:
        image_path = os.path.join(bpath, name)  # then it is a base model image (still call it live for convinience)
    return image_path

def get_subset_of_pairs(all_pair_ids, no):
    random.shuffle(all_pair_ids)
    pair_ids = []
    already_picked_pairs_id = []
    pbar = tqdm(total=no)
    while (len(pair_ids) < no): #could be stuck in endless loop - so just ctrl-c
        rnd_pair_id = random.choice(all_pair_ids)
        if(rnd_pair_id in already_picked_pairs_id):
            continue
        pair_ids.append(rnd_pair_id)  # which is a tuple
        pbar.update(1)
    pbar.close()
    return pair_ids

def createDataForMatchNoMatchMatchabilityComparison(mnm_base_path, output_path, pairs_limit = -1):

    db_live_mnm_path = os.path.join(mnm_base_path, "live/database.db")
    db_live_mnm = COLMAPDatabase.connect(db_live_mnm_path)
    live_model_images_mnm = read_images_binary(os.path.join(mnm_base_path, "live/output_opencv_sift_model/images.bin"))
    image_live_dir_mnm = os.path.join(mnm_base_path, 'live/images/')
    image_base_dir_mnm = os.path.join(mnm_base_path, 'base/images/')

    print("Getting Pairs")
    if (pairs_limit == -1):
        print(f'Getting all pairs..')
        pair_ids = db_live_mnm.execute("SELECT pair_id FROM matches").fetchall()
    else:
        print(f'Getting {pairs_limit} pairs..')
        all_pair_ids = db_live_mnm.execute("SELECT pair_id FROM matches").fetchall()
        pair_ids = get_subset_of_pairs(all_pair_ids, pairs_limit)  # as in paper

    print("Creating data..")
    training_data_db = COLMAPDatabase.create_db_match_no_match_data(os.path.join(output_path, "training_data.db"))
    training_data_db.execute("BEGIN")

    for pair in tqdm(pair_ids):
        pair_id = pair[0]
        img_id_1, img_id_2 = pair_id_to_image_ids(pair_id)

        # when a db image has not been localised ...
        if((img_id_1 not in live_model_images_mnm) or (img_id_2 not in live_model_images_mnm)):
            continue

        img_1_file_name = live_model_images_mnm[img_id_1].name
        img_2_file_name = live_model_images_mnm[img_id_2].name

        img_1_file = cv2.imread(get_full_path(image_live_dir_mnm, image_base_dir_mnm, img_1_file_name))
        img_2_file = cv2.imread(get_full_path(image_live_dir_mnm, image_base_dir_mnm, img_2_file_name))

        pair_data = db_live_mnm.execute("SELECT rows, data FROM matches WHERE pair_id = " + "'" + str(pair_id) + "'").fetchone()
        rows = pair_data[0]

        if(rows < 1): #no matches in this pair, no idea why COLMAP stores it...
            continue

        cols = 2 #for each image
        zero_based_indices = COLMAPDatabase.blob_to_array(pair_data[1], np.uint32).reshape([rows, cols])
        zero_based_indices_left = zero_based_indices[:, 0]
        zero_based_indices_right = zero_based_indices[:, 1]

        keypoints_data_img_1 = get_image_keypoints_data(db_live_mnm, img_id_1)
        keypoints_data_img_2 = get_image_keypoints_data(db_live_mnm, img_id_2)

        keypoints_data_img_1_matched = keypoints_data_img_1[zero_based_indices_left]
        keypoints_data_img_1_unmatched = np.delete(keypoints_data_img_1, zero_based_indices_left, axis=0)

        keypoints_data_img_2_matched = keypoints_data_img_2[zero_based_indices_right]
        keypoints_data_img_2_unmatched = np.delete(keypoints_data_img_2, zero_based_indices_right, axis=0)

        all_kps = [(keypoints_data_img_1_matched, img_1_file),
                   (keypoints_data_img_1_unmatched, img_1_file),
                   (keypoints_data_img_2_matched, img_2_file),
                   (keypoints_data_img_2_unmatched, img_2_file)]

        for i in range(len(all_kps)):
            kps_data = all_kps[i][0]
            img_file = all_kps[i][1]
            if (i % 2) == 0:
                matched = 1 # keypoints_data_img_1_matched, keypoints_data_img_2_matched
            else:
                matched = 0
            for i in range(kps_data.shape[0]): # or kps_decs same
                sample = kps_data[i,:]
                # x, y, octave, angle, size, response, dominantOrientation, green_intensity, matched
                x = sample[0]
                y = sample[1]
                octave = sample[2]
                angle = sample[3]
                size = sample[4]
                response = sample[5]
                dominantOrientation = sample[6]
                green_intensity = img_file[int(y), int(x)][1]  # reverse indexing
                training_data_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                         (float(x),) + (float(y),) + (float(octave),) + (float(angle),) + (float(size),) + (float(response),) + (int(dominantOrientation),) + (float(green_intensity),) + (matched,))

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
mnm_base_path = sys.argv[2] # this is that data generated from format_data_for_match_no_match.py
pairs_limit = int(sys.argv[3])
print("Base (MnM) path: " + mnm_base_path)
output_path = os.path.join(base_path, "match_or_no_match_comparison_data")
os.makedirs(output_path, exist_ok = True)
createDataForMatchNoMatchMatchabilityComparison(mnm_base_path, output_path, pairs_limit)
