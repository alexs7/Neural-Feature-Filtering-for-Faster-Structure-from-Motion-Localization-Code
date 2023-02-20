# Run this after format_data_for_match_no_match.py and get_points_3D_mean_desc_single_model_ml_mnm.py
# if you want to use your pretraining 3D data (not generated here)
# then clear whatever already exists in 'Training Data' (in the build folder of the original C++ tool) and copy paste your tranining data in 'Training Data'
# This file will also create the data for the original C++ tool, and deletes 'Training Data', and 'Training images' folders
# It will run the C++ code "matchornomatch_train". "matchornomatch_train" was compiled separately with different source code.

# The methods in this file do :
# 1 - Generate Training data and Trains RF (Original Tool C++)

# To summarise:
# 1 - Run format_data_for_match_no_match.py on CYENS machine, transfer data back to weatherwax
# 2 - get_points_3D_mean_desc_single_model_ml_mnm.py, on weatherwax
# 3 - Run create_gt_data_for_match_no_match.py to generate GT data once. (This is not needed if you are going to use your own GT 3D data)
# 3 - Run this file, create_training_data_and_train_for_match_no_match.py with some "no_samples" params
# 4 - Look at test_models_on_3D_gt_data.py then and run that

# NOTE: 20/02/2023: You can run file in parallel for each dataset, it will create a folder for each dataset under build/

import os
import shutil
import subprocess
import sys
import numpy as np
from tqdm import tqdm
from database import COLMAPDatabase
from database import pair_id_to_image_ids
from query_image import clear_folder

def get_full_path_live_image(lpath, bpath, name):
    if (('session' in name) == True):
        image_path = os.path.join(lpath, name)
    else:
        image_path = os.path.join(bpath, name)  # then it is a base model image (still call it live for convinience)
    return image_path

def get_image_live_or_base_path(image_name, training_images_path):
    base_image_path = os.path.join(training_images_path, "map/raw_data", image_name)
    if(os.path.exists(base_image_path)):
        return base_image_path
    live_image_path = os.path.join(training_images_path, "query_phone/raw_data", image_name)
    if (os.path.exists(live_image_path)):
        return live_image_path

def get_random_subset_of_pairs(all_pair_ids, no):
    indxs = np.arange(len(all_pair_ids))
    np.random.shuffle(indxs)
    indxs = indxs[0:no]
    return np.array(all_pair_ids)[indxs]

def createTrainingDataForMatchNoMatchMatchabilityComparison(mnm_base_path, mnm_base_code_dir, dataset_name, pairs_limit = -1):
    training_images_folder = f"training_images_{dataset_name}"
    training_data_folder = f"training_data_{dataset_name}"
    trained_model_name = f"trained_model_{dataset_name}"

    # clear folders
    images_for_original_code = os.path.join(mnm_base_code_dir, training_images_folder)  # same name as in MnM code base main.cpp
    clear_folder(images_for_original_code)
    # from the original code "Training Data" will contain the data after the pre-training stage
    # Each time you run the original pre-training (C++) you have to empty the "Training Data" folder, done here
    training_data_csv_for_original_code = os.path.join(mnm_base_code_dir, training_data_folder)
    clear_folder(training_data_csv_for_original_code)

    db_live_mnm_path = os.path.join(mnm_base_path, "live/database.db")
    db_live_mnm = COLMAPDatabase.connect(db_live_mnm_path)
    image_live_dir_mnm = os.path.join(mnm_base_path, 'live/images/')
    image_base_dir_mnm = os.path.join(mnm_base_path, 'base/images/')

    all_pairs_no = len(db_live_mnm.execute("SELECT pair_id FROM two_view_geometries WHERE rows > 10").fetchall())
    print(f"Total numbers of pairs (rows > 10): {all_pairs_no} - Using {pairs_limit} random pairs to generate training data..")

    # pick only pairs that have 10 > rows
    if (pairs_limit == -1):
        print(f'Getting all pairs..')
        pair_ids = db_live_mnm.execute("SELECT pair_id FROM two_view_geometries WHERE rows > 10").fetchall()
    else:
        print(f'Getting {pairs_limit} pairs..')
        all_pair_ids = db_live_mnm.execute("SELECT pair_id FROM two_view_geometries WHERE rows > 10").fetchall()
        pair_ids = get_random_subset_of_pairs(all_pair_ids, pairs_limit)  # as in paper

    print(f"Copying live images to folder \"Training images\"")
    file_index = 1
    for i in tqdm(range(len(pair_ids))):
        pair_id = pair_ids[i][0]
        img_id_1, img_id_2 = pair_id_to_image_ids(pair_id)
        assert img_id_1 != img_id_2

        # copy the image files too in the MnM original code data dir - to use later with the original code
        # The original code, C++, does all the pre-training, extract features etc etc..
        img_1_name = db_live_mnm.execute("SELECT name FROM images WHERE image_id = " + "'" + str(img_id_1) + "'").fetchone()[0]
        shutil.copyfile(get_full_path_live_image(image_live_dir_mnm, image_base_dir_mnm, img_1_name),
                        os.path.join(images_for_original_code, f"image_{'%010d' % file_index}.jpg"))
        img_2_name = db_live_mnm.execute("SELECT name FROM images WHERE image_id = " + "'" + str(img_id_2) + "'").fetchone()[0]
        shutil.copyfile(get_full_path_live_image(image_live_dir_mnm, image_base_dir_mnm, img_2_name),
                        os.path.join(images_for_original_code, f"image_{'%010d' % (file_index + 1)}.jpg"))
        file_index += 2

    print("Pairs generated! - This is the training data you need for the original C++ code.")

    print(f"Training the original C++ code.. for {pairs_limit} pairs")
    matchornomatch_train = ["./matchornomatch_train", training_images_folder, training_data_folder, trained_model_name]
    print(f"Running command: {' '.join(matchornomatch_train)}")
    subprocess.check_call(matchornomatch_train, cwd=mnm_base_code_dir)

    # cp Trained model to data_path
    shutil.copyfile(os.path.join(mnm_base_code_dir, trained_model_name), os.path.join(mnm_base_path, f"{trained_model_name}_pairs_no_{pairs_limit}.xml"))

    print("Done!")

def createTrainingDataForMatchNoMatchMatchabilityComparisonLaMAR(mnm_base_path, training_images_path, mnm_base_code_dir, dataset_name, pairs_limit = -1):
    training_images_folder = f"training_images_{dataset_name}"
    training_data_folder = f"training_data_{dataset_name}"
    trained_model_name = f"trained_model_{dataset_name}"

    # clear folders
    images_for_original_code = os.path.join(mnm_base_code_dir, training_images_folder)  # same name as in MnM code base main.cpp
    clear_folder(images_for_original_code)
    # from the original code "Training Data" will contain the data after the pre-training stage
    # Each time you run the original pre-training (C++) you have to empty the "Training Data" folder, done here
    training_data_csv_for_original_code = os.path.join(mnm_base_code_dir, training_data_folder)
    clear_folder(training_data_csv_for_original_code)

    db_live_mnm_path = os.path.join(mnm_base_path, "live/database.db")
    db_live_mnm = COLMAPDatabase.connect(db_live_mnm_path)

    all_pairs_no = len(db_live_mnm.execute("SELECT pair_id FROM two_view_geometries WHERE rows > 10").fetchall())
    print(f"Total numbers of pairs (rows > 10): {all_pairs_no} - Using {pairs_limit} random pairs to generate training data..")

    # pick only pairs that have 10 > rows
    if (pairs_limit == -1):
        print(f'Getting all pairs..')
        pair_ids = db_live_mnm.execute("SELECT pair_id FROM two_view_geometries WHERE rows > 10").fetchall()
    else:
        print(f'Getting {pairs_limit} pairs..')
        all_pair_ids = db_live_mnm.execute("SELECT pair_id FROM two_view_geometries WHERE rows > 10").fetchall()
        pair_ids = get_random_subset_of_pairs(all_pair_ids, pairs_limit)  # as in paper

    print(f"Copying live images to folder \"Training images\"")
    file_index = 1
    for i in tqdm(range(len(pair_ids))):
        pair_id = pair_ids[i][0]
        img_id_1, img_id_2 = pair_id_to_image_ids(pair_id)
        assert img_id_1 != img_id_2

        # copy the image files too in the MnM original code data dir - to use later with the original code
        # The original code, C++, does all the pre-training, extract features etc etc..
        img_1_name = db_live_mnm.execute("SELECT name FROM images WHERE image_id = " + "'" + str(img_id_1) + "'").fetchone()[0]
        img_1_source_path = get_image_live_or_base_path(img_1_name, training_images_path)
        shutil.copyfile(img_1_source_path, os.path.join(images_for_original_code, f"image_{'%010d' % file_index}.jpg"))
        img_2_name = db_live_mnm.execute("SELECT name FROM images WHERE image_id = " + "'" + str(img_id_2) + "'").fetchone()[0]
        img_2_source_path = get_image_live_or_base_path(img_2_name, training_images_path)
        shutil.copyfile(img_2_source_path, os.path.join(images_for_original_code, f"image_{'%010d' % (file_index + 1)}.jpg"))
        file_index += 2

    print("Pairs generated! - This is the training data you need for the original C++ code.")

    print(f"Training the original C++ code.. for {pairs_limit} pairs")
    matchornomatch_train = ["./matchornomatch_train", training_images_folder, training_data_folder, trained_model_name]
    print(f"Running command: {' '.join(matchornomatch_train)}")
    subprocess.check_call(matchornomatch_train, cwd=mnm_base_code_dir)

    # cp Trained model to data_path
    shutil.copyfile(os.path.join(mnm_base_code_dir, trained_model_name), os.path.join(mnm_base_path, f"{trained_model_name}_pairs_no_{pairs_limit}.xml"))

    print("Done!")

root_path = "/media/iNicosiaData/engd_data/"
# This will contain images and csv files to use for the MnM code! (relative path)
mnm_base_source_code_dir = "code_to_compare/Match-or-no-match-Keypoint-filtering-based-on-matching-probability/build/"

dataset = sys.argv[1]

if(dataset == "HGE" or dataset == "CAB" or dataset == "LIN"):
    base_path = os.path.join(root_path, "lamar", f"{dataset}_colmap_model/models_for_match_no_match")
    print(f"Base path: {base_path}")
    live_images_path = os.path.join(root_path, "lamar", f"{dataset}/sessions/")  # this is the live images and abse images parent folder - the choice happens in createTrainingDataForMatchNoMatchMatchabilityComparisonLaMAR
    for arg in sys.argv[2:]:
        print("Doing " + arg + " pairs..")
        pairs_limit = int(arg)  # how many samples to test, in the original paper it was ~ 150, but they mention individual images not pairs
        createTrainingDataForMatchNoMatchMatchabilityComparisonLaMAR(base_path, live_images_path, mnm_base_source_code_dir, dataset, pairs_limit)

if(dataset == "CMU"):
    slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                    "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    for slice_name in slices_names:
        base_path = os.path.join(root_path, "cmu", f"{slice_name}", "exmaps_data", "models_for_match_no_match")
        print("Base path: " + base_path)
        for arg in sys.argv[2:]:
            print("Doing " + arg + " pairs..")
            pairs_limit = int(arg)  # how many samples to test, in the original paper it was ~ 150, but they mention individual images not pairs
            createTrainingDataForMatchNoMatchMatchabilityComparison(base_path, mnm_base_source_code_dir, slice_name, pairs_limit)

if(dataset == "RetailShop"):
    base_path = os.path.join(root_path, "retail_shop", "slice1", "models_for_match_no_match")
    print("Base path: " + base_path)
    for arg in sys.argv[2:]:
        print("Doing " + arg + " pairs..")
        pairs_limit = int(arg)  # how many samples to test, in the original paper it was ~ 150, but they mention individual images not pairs
        createTrainingDataForMatchNoMatchMatchabilityComparison(base_path, mnm_base_source_code_dir, dataset, pairs_limit)