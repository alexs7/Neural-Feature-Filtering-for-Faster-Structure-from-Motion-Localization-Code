# Run this after format_data_for_match_no_match.py
import csv
import glob
import os
import shutil
import sys
import cv2
from database import COLMAPDatabase
from database import pair_id_to_image_ids
import numpy as np
import random
from tqdm import tqdm
from point3D_loader import read_points3d_default
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_image_id, get_image_name_only


# similar code is used in feature_matching_generator_ML_comparison_models.py
def get_image_data(db, points3D, images, img_id, img_file):
    image = images[img_id] #only localised images
    kp_db_row = db.execute("SELECT rows, cols, data, dominantOrientations FROM keypoints WHERE image_id = " + "'" + str(img_id) + "'").fetchone()
    cols = kp_db_row[1]
    rows = kp_db_row[0]

    assert (image.xys.shape[0] == image.point3D_ids.shape[0] == rows)  # just for my sanity
    # x, y, octave, angle, size, response
    kp_data = COLMAPDatabase.blob_to_array(kp_db_row[2], np.float32)
    kp_data = kp_data.reshape([rows, cols])
    dominantOrientations = COLMAPDatabase.blob_to_array(kp_db_row[3], np.uint8)
    dominantOrientations = dominantOrientations.reshape([rows, 1])

    matched_values = [] #for each keypoint (x,y)/desc same thing
    green_intensities = [] #for each keypoint (x,y)/desc same thing

    for i in range(image.xys.shape[0]):  # can loop through descs or img_data.xys - same thing
        current_point3D_id = image.point3D_ids[i]
        x = image.xys[i][0]
        y = image.xys[i][1]
        if (current_point3D_id == -1):  # means feature is unmatched
            matched = 0
            green_intensity = img_file[int(y), int(x)][1] # reverse indexing
        else:
            # this is to make sure that xy belong to the right pointd3D
            assert i in points3D[current_point3D_id].point2D_idxs
            matched = 1
            green_intensity = img_file[int(y), int(x)][1] # reverse indexing
        matched_values.append(matched)
        green_intensities.append(green_intensity)

    matched_values = np.array(matched_values).reshape(rows, 1)
    green_intensities = np.array(green_intensities).reshape(rows, 1)

    return np.c_[kp_data, dominantOrientations, green_intensities, matched_values]

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

def createDataForMatchNoMatchMatchabilityComparison(mnm_base_path, base_path, output_path, files_for_original_code, pairs_limit = -1):

    db_live_mnm_path = os.path.join(mnm_base_path, "live/database.db")
    db_live_mnm = COLMAPDatabase.connect(db_live_mnm_path)
    live_model_images_mnm = read_images_binary(os.path.join(mnm_base_path, "live/output_opencv_sift_model/images.bin"))
    live_points_3D_mnm = read_points3d_default(os.path.join(mnm_base_path, "live/output_opencv_sift_model/points3D.bin"))
    image_live_dir_mnm = os.path.join(mnm_base_path, 'live/images/')
    image_base_dir_mnm = os.path.join(mnm_base_path, 'base/images/')

    if (pairs_limit == -1):
        print(f'Getting all pairs..')
        pair_ids = db_live_mnm.execute("SELECT pair_id FROM matches").fetchall()
    else:
        print(f'Getting {pairs_limit} pairs..')
        all_pair_ids = db_live_mnm.execute("SELECT pair_id FROM matches").fetchall()
        pair_ids = get_subset_of_pairs(all_pair_ids, pairs_limit)  # as in paper

    print("Creating data..")
    training_and_test_data_db = COLMAPDatabase.create_db_match_no_match_data(os.path.join(output_path, "training_data.db"))
    training_and_test_data_db.execute("BEGIN")

    # This is to avoid cases such as (1.0, 2), (1.0, 3) .. to double adding 1.0 etc
    # and also (3.0, 1) when its reveresed if it can be
    # we need two lists here
    all_images_id = []

    for i in tqdm(range(len(pair_ids))):
        pair_id = pair_ids[i][0]
        pair_data = db_live_mnm.execute("SELECT rows, data FROM matches WHERE pair_id = " + "'" + str(pair_id) + "'").fetchone()
        rows = pair_data[0]
        if (rows < 1):  # no matches in this pair, no idea why COLMAP stores it...
            continue
        img_id_1, img_id_2 = pair_id_to_image_ids(pair_id)

        img_1_name = db_live_mnm.execute("SELECT name FROM images WHERE image_id = " + "'" + str(img_id_1) + "'").fetchone()[0]
        img_2_name = db_live_mnm.execute("SELECT name FROM images WHERE image_id = " + "'" + str(img_id_2) + "'").fetchone()[0]
        # copy the image files too in the MnM original code data dir - to use later with the original code
        shutil.copyfile(get_full_path(image_live_dir_mnm, image_base_dir_mnm, img_1_name), os.path.join(files_for_original_code, f"{i}.jpg"))
        shutil.copyfile(get_full_path(image_live_dir_mnm, image_base_dir_mnm, img_2_name), os.path.join(files_for_original_code, f"{i+1}.jpg"))

        if(img_id_1 not in all_images_id):
            all_images_id.append(img_id_1)
        if (img_id_2 not in all_images_id):
            all_images_id.append(img_id_2)

    print(f"Size of all_images_id: {len(all_images_id)}")
    print(f"Size of live_model_images_mnm: {len(live_model_images_mnm)}")

    # This code follows similar structure from App.cpp (the c++ paper code)
    print("Inserting Train Data in db and creating a .csv file for each image for the original code..")

    failure_images = 0
    for img_id in tqdm(all_images_id):
        # when a db image has not been localised ...
        if((img_id not in live_model_images_mnm)):
            failure_images = failure_images + 1
            continue

        img_file_name = live_model_images_mnm[img_id].name
        img_file = cv2.imread(get_full_path(image_live_dir_mnm, image_base_dir_mnm, img_file_name))
        training_data_img = get_image_data(db_live_mnm, live_points_3D_mnm, live_model_images_mnm, img_id, img_file)

        image_name_plain = get_image_name_only(img_file_name) #deals with session_6/image...
        with open(os.path.join(files_for_original_code, f"{image_name_plain}.csv"), 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            # insert image's data in db and csv
            for i in range(len(training_data_img)):
                sample = training_data_img[i, :]
                x = sample[0]
                y = sample[1]
                octave = sample[2]
                angle = sample[3]
                size = sample[4]
                response = sample[5]
                dominantOrientation = sample[6]
                green_intensity = sample[7]
                matched = sample[8] #can use astype(np.int64) here
                testSample = 0
                # xs, ys, octaves, angles, sizes, responses, dominantOrientations, greenInt
                training_and_test_data_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (float(x),) + (float(y),) + (float(octave),) + (float(angle),) + (float(size),) + (float(response),) + (int(dominantOrientation),) + (
                               float(green_intensity),) + (matched,) + (testSample,) + (img_id,))
                # x, y coordinates, octave, angle, size, response, green color value and dominant orientations, matched (from: https://github.com/AlexandraPapadaki/Match-or-no-match-Keypoint-filtering-based-on-matching-probability)
                csv_row = [x,y,octave,angle,size,response,green_intensity,dominantOrientation,matched]
                writer.writerow(csv_row)

    print(f"Images from db that were NOT localised (in the .bin file, i.e no ground truth data): {failure_images}")
    # add test data too (gt = query as we know)
    print("Inserting Test Data..")
    query_images_path = os.path.join(base_path, "gt/query_name.txt") #these are the same anw
    query_images_names = load_images_from_text_file(query_images_path)
    db_gt_mnm_path = os.path.join(mnm_base_path, "gt/database.db") #openCV db + extra MnM data
    db_gt_mnm = COLMAPDatabase.connect(db_gt_mnm_path)  # remember this database holds the OpenCV descriptors
    query_images_bin_path_mnm = os.path.join(mnm_base_path, "gt/output_opencv_sift_model/images.bin")
    localised_query_images_names_mnm = get_localised_image_by_names(query_images_names, query_images_bin_path_mnm)
    image_gt_dir_mnm = os.path.join(base_path, 'gt/images/') # or mnm_base_path should be the same
    gt_points_3D_mnm = read_points3d_default(os.path.join(mnm_base_path, "gt/output_opencv_sift_model/points3D.bin"))
    gt_model_images_mnm = read_images_binary(query_images_bin_path_mnm)

    for i in tqdm(range(len(localised_query_images_names_mnm))):
        img_name = localised_query_images_names_mnm[i]
        image_gt_path = os.path.join(image_gt_dir_mnm, img_name)
        qt_image_file = cv2.imread(image_gt_path)  # no need for cv2.COLOR_BGR2RGB here as G is in the middle anw
        image_id = int(get_image_id(db_gt_mnm, img_name))
        test_data_img = get_image_data(db_gt_mnm, gt_points_3D_mnm, gt_model_images_mnm, image_id, qt_image_file)

        for i in range(len(test_data_img)):
            sample = test_data_img[i, :]
            x = sample[0]
            y = sample[1]
            octave = sample[2]
            angle = sample[3]
            size = sample[4]
            response = sample[5]
            dominantOrientation = sample[6]
            green_intensity = sample[7]
            matched = sample[8] #can use astype(np.int64) here
            testSample = 1
            # xs, ys, octaves, angles, sizes, responses, dominantOrientations, greenInt
            training_and_test_data_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (float(x),) + (float(y),) + (float(octave),) + (float(angle),) + (float(size),) + (float(response),) + (int(dominantOrientation),) + (
                           float(green_intensity),) + (matched,) + (testSample,) + (image_id,))

    print("Committing..")
    training_and_test_data_db.commit()
    print("Generating Stats (only for training data)..")
    stats = training_and_test_data_db.execute("SELECT * FROM data WHERE testSample = 0").fetchall()
    matched = training_and_test_data_db.execute("SELECT * FROM data WHERE matched = 1 AND testSample = 0").fetchall()
    unmatched = training_and_test_data_db.execute("SELECT * FROM data WHERE matched = 0 AND testSample = 0").fetchall()

    print("Total samples: " + str(len(stats)))
    print("Total matched samples: " + str(len(matched)))
    print("Total unmatched samples: " + str(len(unmatched)))
    print("% of matched samples: " + str(len(matched) * 100 / len(stats)))

    print("Done!")

base_path = sys.argv[1]
mnm_base_path = sys.argv[2] # this is that data generated from format_data_for_match_no_match.py
pairs_limit = int(sys.argv[3])
print("Base (MnM) path: " + mnm_base_path)
# will store the training_data and test_data in base_path for convenience
output_path = os.path.join(base_path, "match_or_no_match_comparison_data")
# This will contain images and csv files to use for the MnM code!
files_for_original_code = os.path.join(output_path, "files_for_original_code")
os.makedirs(output_path, exist_ok = True)
os.makedirs(files_for_original_code, exist_ok = True)
createDataForMatchNoMatchMatchabilityComparison(mnm_base_path, base_path, output_path, files_for_original_code, pairs_limit)
