# Run this after format_data_for_match_no_match.py and get_points_3D_mean_desc_single_model_ml_mnm.py
# You need the "all_images" folder for this script to work, from format_data_for_match_no_match.py.
# The "all_images" folder is in each dataset folder.
# This file will create the data for the original C++ tool, and deletes 'Training Data', and 'Training images' folders
# It will run the C++ code "matchornomatch_train". "matchornomatch_train" was compiled separately with different source code.

# The methods in this file do :
# 1 - Generate Training data and Trains RF (Original Tool C++)

# To summarise:
# 1 - Run format_data_for_match_no_match.py on CYENS machine, transfer data back to weatherwax
# 2 - get_points_3D_mean_desc_single_model_ml_mnm.py, on weatherwax (not needed for this file)
# 3 - Run this file, create_training_data_and_train_for_match_no_match.py with some "pairs_limit" params
# 4 - Look at test_models_on_3D_gt_data.py then and run that

# NOTE: 20/02/2023: You can run file in parallel for each dataset, it will create a folder for each dataset under build/
# NOTE: 03/03/2023: This file does not depend on format_data_for_match_no_match.py other than the "all_images" folder (which is created by format_data_for_match_no_match.py)
# the ids of the images are the same in the live db and the mnm db, so we can use the live db to get the pairs

import os
import shutil
import subprocess
import sys
import numpy as np
from tqdm import tqdm
from database import COLMAPDatabase
from database import pair_id_to_image_ids
from query_image import clear_folder

def get_random_subset_of_pairs(all_pair_ids, no):
    indxs = np.arange(len(all_pair_ids))
    np.random.shuffle(indxs)
    indxs = indxs[0:no]
    return np.array(all_pair_ids)[indxs]

def createTrainingDataForMatchNoMatchMatchabilityComparison(mnm_path, original_live_db_path, all_images_path, mnm_base_code_dir, pairs_limit = -1):
    training_images_folder = os.path.join(mnm_path, f"training_images")
    training_data_folder = os.path.join(mnm_path, f"training_data") #will contain the data after the pre-training stage, that will be fed to the RF
    trained_model_name = os.path.join(mnm_path, f"trained_model")

    # clear folders from previous runs
    clear_folder(training_images_folder)
    clear_folder(training_data_folder)

    original_live_db = COLMAPDatabase.connect(original_live_db_path) #can just use the live db to get the pairs (same as mnm db)

    rows_no = 10
    print(f"Getting all pairs (rows > {rows_no}) from the original live database..")
    all_pairs_no = len(original_live_db.execute(f"SELECT pair_id FROM two_view_geometries WHERE rows > {rows_no}").fetchall())
    print(f"Total numbers of pairs (rows > 10): {all_pairs_no} - Using {pairs_limit} random pairs to generate training data..")

    # pick only pairs that have 10 > rows
    if (pairs_limit == -1):
        print(f'Getting all pairs..')
        pair_ids = original_live_db.execute("SELECT pair_id FROM two_view_geometries WHERE rows > 10").fetchall()
    else:
        print(f'Getting {pairs_limit} pairs..')
        all_pair_ids = original_live_db.execute("SELECT pair_id FROM two_view_geometries WHERE rows > 10").fetchall()
        pair_ids = get_random_subset_of_pairs(all_pair_ids, pairs_limit)  # as in paper

    print(f"Copying live images to folder \"Training images\"")
    file_index = 1
    for i in tqdm(range(len(pair_ids))):
        pair_id = pair_ids[i][0]
        img_id_1, img_id_2 = pair_id_to_image_ids(pair_id)
        assert img_id_1 != img_id_2

        # copy the image files too in the MnM original code data dir - to use later with the original code
        # The original code, C++, does all the pre-training, extract features etc etc..
        img_1_name = original_live_db.execute("SELECT name FROM images WHERE image_id = " + "'" + str(img_id_1) + "'").fetchone()[0]
        img_1_source_path = os.path.join(all_images_path, img_1_name)
        shutil.copyfile(img_1_source_path, os.path.join(training_images_folder, f"image_{'%010d' % file_index}.jpg"))
        img_2_name = original_live_db.execute("SELECT name FROM images WHERE image_id = " + "'" + str(img_id_2) + "'").fetchone()[0]
        img_2_source_path = os.path.join(all_images_path, img_2_name)
        shutil.copyfile(img_2_source_path, os.path.join(training_images_folder, f"image_{'%010d' % (file_index + 1)}.jpg"))
        file_index += 2

    print("Pairs generated! - This is the training data you need for the original C++ code.")

    print(f"Training the original C++ code.. for {pairs_limit} pairs")
    # Using absolute paths for the C++ code
    matchornomatch_train = ["./matchornomatch_train", training_images_folder, training_data_folder, trained_model_name]
    print(f"Running command: {' '.join(matchornomatch_train)}")
    subprocess.check_call(matchornomatch_train, cwd=mnm_base_code_dir)

    # cp Trained model to data_path
    shutil.copyfile(os.path.join(mnm_base_code_dir, trained_model_name), os.path.join(mnm_path, f"{trained_model_name}_pairs_no_{pairs_limit}.xml"))

    print("Done!")

root_path = "/media/iNicosiaData/engd_data/"
# This will contain images and csv files to use for the MnM code! (relative path)
mnm_base_source_code_dir = "code_to_compare/Match-or-no-match-Keypoint-filtering-based-on-matching-probability/build/"

dataset = sys.argv[1]

# For each dataset I use hte "all_images" folder, which contains all images from each dataset that was generated by format_data_for_match_no_match.py
# The images will be moved from there to the training_images folder in the MnM code base
# I am also using the original live database.db from each dataset, to get the pairs as they are identical to the pairs in the mnm gt database.
# In format_data_for_match_no_match.py I am using the same pairs as in the original models

if(dataset == "HGE" or dataset == "CAB" or dataset == "LIN"):
    mnm_path = os.path.join(root_path, "lamar", f"{dataset}_colmap_model/models_for_match_no_match")
    print(f"Base path: {mnm_path}")
    original_live_db_path = os.path.join(root_path, "lamar", f"{dataset}_colmap_model", "live", "database.db")
    images_path = os.path.join(mnm_path, "all_images")  # all images from format_data_for_match_no_match.py
    pairs_no = sys.argv[2]
    print("Doing " + pairs_no + " pairs..")
    pairs_limit = int(pairs_no)  # how many samples to test, in the original paper it was ~ 150, but they mention individual images not pairs
    createTrainingDataForMatchNoMatchMatchabilityComparison(mnm_path, original_live_db_path, images_path, mnm_base_source_code_dir, pairs_limit)

if(dataset == "CMU"):
    slices_names = ["slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    for slice_name in slices_names:
        mnm_path = os.path.join(root_path, "cmu", f"{slice_name}", "exmaps_data", "models_for_match_no_match")
        print("Base path: " + mnm_path)
        original_live_db_path = os.path.join(root_path, "cmu", f"{slice_name}", "exmaps_data", "live", "database.db")
        images_path = os.path.join(mnm_path, "all_images")  # all images from format_data_for_match_no_match.py
        pairs_no = sys.argv[2]
        print("Doing " + pairs_no + " pairs..")
        pairs_limit = int(pairs_no)  # how many samples to test, in the original paper it was ~ 150, but they mention individual images not pairs
        dataset_name = f"{dataset}_{slice_name}"
        createTrainingDataForMatchNoMatchMatchabilityComparison(mnm_path, original_live_db_path, images_path, mnm_base_source_code_dir, pairs_limit)

if(dataset == "RetailShop"):
    mnm_path = os.path.join(root_path, "retail_shop", "slice1", "models_for_match_no_match")
    print("Base path: " + mnm_path)
    original_live_db_path = os.path.join(root_path, "retail_shop", "slice1", "live", "database.db")
    images_path = os.path.join(mnm_path, "all_images")  # all images from format_data_for_match_no_match.py
    pairs_no = sys.argv[2]
    print("Doing " + pairs_no + " pairs..")
    pairs_limit = int(pairs_no)  # how many samples to test, in the original paper it was ~ 150, but they mention individual images not pairs
    createTrainingDataForMatchNoMatchMatchabilityComparison(mnm_path, original_live_db_path, images_path, mnm_base_source_code_dir, pairs_limit)