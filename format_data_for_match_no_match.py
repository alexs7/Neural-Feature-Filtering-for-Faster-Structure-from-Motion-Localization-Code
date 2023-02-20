# Run this file before creating the data for MnM - (2020) paper, and run it on the CYENS machine.
# This file will aim to create the data for the RF model from Match no Match, MnM - (2020) paper
# It will extract OpenCV SIFT features and insert them in colmap's database and run the triangulator again for the base only.
# For the live and gt model, I just run the colmap's image_registrator, same as my first publication.
# The base model is triangulated using the already known camera poses from the original model
# I replace the old data, keypoints descriptors, and keep the poses (check COLMAP FAQ).
# You will need to run this on the CYENS machine as it has pycolmap and colmap installed - because of docker I can't run them on Bath Uni

# NOTE: 22/12/2022 The data can be kept on the CYENS machine. All the data is there now.

# TODO: use the colmap exhaustive_matcher with --SiftMatching.use_gpu 1 for the query stage, if images are too many then no
# TODO: or train your own as here: https://github.com/colmap/colmap/issues/866

import glob
import os
import random
import shutil
import sys
import cv2
import numpy as np
import pycolmap
from tqdm import tqdm
import colmap
from database import COLMAPDatabase
from helper import remove_folder_safe
from query_image import get_all_images_names_from_db, get_image_id

MULTIPLIER = 4 #increase this to increase the number of features extracted and get more images localised in live and gt

# The more you increase the values the more images will localise from live and gt
# The randomness is to simulate the COLMAP feature extraction 800 query / 2000 recon. used in previous papers
query_features_limit = MULTIPLIER * random.randint(700, 900)
reconstr_features_limit = MULTIPLIER * random.randint(1900, 2200)

def empty_points_3D_txt_file(path):
    open(path, 'w').close()

def arrange_images_txt_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    with open(path, "w") as f:
        for line in tqdm(lines):
            if "#" in line or ".jpg" in line:
                f.write(line)
            else:
                f.write("\n")

def create_new_query_image_names_file(model_path):
    # just rewrite existing one
    new_query_image_names_file_path = os.path.join(model_path, f'query_name.txt') #new will contain absolute paths
    model_images_path = os.path.join(model_path, 'images')

    if("base" in model_path):
        search_path = '*' #images/*.jpg
    else:
        search_path = '/**/*' #images/session_*/*.jpg

    with open(new_query_image_names_file_path, 'w') as f:
        for filename in glob.glob(model_images_path + search_path):
            f.write(f"{filename}\n")

    return new_query_image_names_file_path

def countDominantOrientations(keypoints): #13/02/2023 refactored to be faster
    domOrientations = np.ones([len(keypoints), 1])
    x1 = np.array([kp.pt[0] for kp in keypoints])
    x2 = np.array([kp.pt[0] for kp in keypoints])
    y1 = np.array([kp.pt[1] for kp in keypoints])
    y2 = np.array([kp.pt[1] for kp in keypoints])

    x1x1 = x1[:, np.newaxis]
    x2x2 = x2[np.newaxis, :]

    y1y1 = y1[:, np.newaxis]
    y2y2 = y2[np.newaxis, :]

    x_comp = np.abs(x1x1 - x2x2)
    y_comp = np.abs(y1y1 - y2y2)
    dist = x_comp + y_comp
    np.fill_diagonal(dist, -1) #set it to -1, so we can get the element with the zero value
    domOrientations[np.where(dist == 0)[0]] = 2
    return domOrientations

def image_is_base_and_has_no_keypoints(image_id, base_model_images):
    return (int(image_id) in base_model_images.keys() and len(base_model_images[int(image_id)].xys) == 0)

def prepare_all_data_for_match_no_match(base_path, original_path, dataset=None, doing_lamar=False):
    # MnM model paths
    model_base_path = os.path.join(base_path, "base")
    model_live_path = os.path.join(base_path, "live")
    model_gt_path = os.path.join(base_path, "gt")

    # MnM database paths
    base_db_path = os.path.join(model_base_path, 'database.db')
    live_db_path = os.path.join(model_live_path, 'database.db')
    qt_db_path = os.path.join(model_gt_path, 'database.db')

    # NOTE 13/02/2023:
    # Since I am running this on CYENS now I will copy base to the folder "base_path"
    print(f"Copying base, only and register live and gt images on top of it..")
    shutil.copytree(os.path.join(original_path, "base",), model_base_path, dirs_exist_ok=True)
    shutil.copytree(os.path.join(original_path, "live",), model_live_path, dirs_exist_ok=True)
    shutil.copytree(os.path.join(original_path, "gt",), model_gt_path, dirs_exist_ok=True)

    if(doing_lamar == True):
        images_base_path = os.path.join("/media/iNicosiaData/engd_data/lamar/", dataset, "sessions", "map", "raw_data")
        images_live_path = os.path.join("/media/iNicosiaData/engd_data/lamar/", dataset, "sessions", "query_phone", "raw_data")
        images_gt_path = os.path.join("/media/iNicosiaData/engd_data/lamar/", dataset, "sessions", "query_val_phone", "raw_data")
    else:
        images_base_path = os.path.join(model_base_path, "images")
        images_live_path = os.path.join(model_live_path, "images")
        images_gt_path = os.path.join(model_gt_path, "images")

    sift = cv2.SIFT_create()

    # look at cmu_sparse_reconstuctor.py, for help
    # Note: use images names from database to locate them for opencv feature extraction

    # 1 - replace base model features with openCV sift (including matches too)
    # "empty model" can be deleted at the end
    manually_created_model_txt_path = os.path.join(model_base_path, 'empty_model_for_triangulation_txt')  # the "empty model" that will be used to create "opencv_sift_model"
    os.makedirs(manually_created_model_txt_path, exist_ok=True)

    # The original model is from ExMaps, get_lamar.py. It has been used to run benchmarks, so it is reliable enough to use (it was copied over).
    colmap_model_path = os.path.join(model_base_path, 'model') #the original model is in here
    reconstruction = pycolmap.Reconstruction(colmap_model_path) #loading the original model from Exmaps

    # set up files as stated online in COLMAP's faq
    # export model to txt
    print("Exporting model to txt..")
    reconstruction.write_text(manually_created_model_txt_path) #this will create the files: cameras.txt, images.txt, points3D.txt
    points_3D_file_txt_path = os.path.join(manually_created_model_txt_path, 'points3D.txt')
    images_file_txt_path = os.path.join(manually_created_model_txt_path, 'images.txt')
    empty_points_3D_txt_file(points_3D_file_txt_path) #as in COLMAP's faq
    arrange_images_txt_file(images_file_txt_path) #as in COLMAP's faq

    base_db = COLMAPDatabase.connect(base_db_path)
    image_names = get_all_images_names_from_db(base_db) #at this point its just base images

    if(base_db.dominant_orientations_column_exists() == False):
        base_db.add_dominant_orientations_column()
        base_db.commit() #we need to commit here

    print("Extracting data from images (base)..")
    for image_name in tqdm(image_names):
        image_file_path = os.path.join(images_base_path, image_name)
        img = cv2.imread(image_file_path)
        kps_plain = []
        kps, des = sift.detectAndCompute(img,None)
        if(des is None): #this happens with textureless images, such as 71411677979.jpg
            continue
        # as in paper ~2000 for map, ~800 for query
        # this might lead to same number of features in the database for the base images.
        # For example if more than reconstr_features_limit are detected for multiple images
        # then they will be reconstr_features_limit for multiple images
        idxs = np.arange(len(kps))
        np.random.shuffle(idxs)
        rnd_idxs = idxs[0:reconstr_features_limit] #random idxs
        kps = np.array(kps)
        kps = kps[rnd_idxs] #replace with random
        des = np.array(des)
        des = des[rnd_idxs] #replace with random
        dominant_orientations = countDominantOrientations(kps)

        kps_plain += [[kps[i].pt[0], kps[i].pt[1], kps[i].octave, kps[i].angle, kps[i].size, kps[i].response] for i in range(len(kps))]
        kps_plain = np.array(kps_plain)
        image_id = get_image_id(base_db, image_name)
        base_db.replace_keypoints(image_id, kps_plain, dominant_orientations)
        base_db.replace_descriptors(image_id, des)
        base_db.commit()

    print("Deleting previous matches..")
    base_db.delete_all_matches()
    base_db.delete_all_two_view_geometries()
    base_db.commit()

    print("Matching -> at base model")
    colmap.vocab_tree_matcher(base_db_path)

    # 2 - triangulate the base model -> base opencv sift model
    opencv_sift_base_model_path = os.path.join(model_base_path, 'output_opencv_sift_model')
    colmap.point_triangulator(base_db_path, images_base_path, manually_created_model_txt_path, opencv_sift_base_model_path)

    print('Base Model done!')
    print()

    print("Copying original base .db over..")
    shutil.copyfile(base_db_path, live_db_path) #overwrite live db with base db

    # 3 - replace the live database's features with opencv sift features
    # live db at this point is the same as base db
    live_db = COLMAPDatabase.connect(live_db_path)
    original_live_db = COLMAPDatabase.connect(os.path.join(original_path, "live", "database.db"))
    # The query_name.txt file will contain the names of the live images only as it reads their name from the folder (from Exmaps)
    # The db will contain all of them BUT the model will contain less as NOT all localise. Same with gt images.
    query_live_images_txt_path = os.path.join(model_live_path, "query_name.txt")
    image_names = np.loadtxt(query_live_images_txt_path, dtype=str) #only live images

    print("Extracting data from images (live only - ignoring the ones base as we already extracted data from them)..")

    for image_name in tqdm(image_names):
        # at this point we are looking at a live image_id only.
        img = cv2.imread(os.path.join(images_live_path, image_name))
        kps_plain = []
        kps, des = sift.detectAndCompute(img,None)
        if (des is None):  # this happens with textureless images, such as 71411677979.jpg
            continue
        # as in paper ~2000 for map, ~800 for query
        idxs = np.arange(len(kps))
        np.random.shuffle(idxs)
        # TODO: pick the same number of keypoints as in the original live db
        rnd_idxs = idxs[0:query_features_limit]  # random idxs query one here
        kps = np.array(kps)
        kps = kps[rnd_idxs]  # replace with random
        des = np.array(des)
        des = des[rnd_idxs]  # replace with random
        dominant_orientations = countDominantOrientations(kps)

        kps_plain += [[kps[i].pt[0], kps[i].pt[1], kps[i].octave, kps[i].angle, kps[i].size, kps[i].response] for i in range(len(kps))]
        kps_plain = np.array(kps_plain)

        # copy from original live db (note the fetching by image name and cam_id)
        img_details = original_live_db.execute("SELECT * FROM images WHERE name = " + "'" + str(image_name) + "'").fetchone()
        live_db.execute("INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (img_details[0],) + (img_details[1],) + (img_details[2],) + (img_details[3],) + (img_details[4],) + (img_details[5],) + (img_details[6],) + (img_details[7],) + (img_details[8],) + (img_details[9],))
        cam_details = original_live_db.execute("SELECT * FROM cameras WHERE camera_id = " + "'" + str(img_details[2]) + "'").fetchone()
        # INSERT OR IGNORE is used for CMU and Retail datasets, as they have base, live and gt images with the same three camera_ids
        live_db.execute("INSERT OR IGNORE INTO cameras VALUES (?, ?, ?, ?, ?, ?)", (cam_details[0],) + (cam_details[1],) + (cam_details[2],) + (cam_details[3],) + (cam_details[4],) + (cam_details[5],))
        image_id = img_details[0]
        live_db.insert_keypoints(image_id, kps_plain, dominant_orientations)
        live_db.insert_descriptors(image_id, des)
        live_db.commit()

    print("Matching -> at live model")
    colmap.vocab_tree_matcher(live_db_path, query_live_images_txt_path)

    # 4 - register the new live images against the base opencv sift model
    opencv_sift_live_model_path = os.path.join(model_live_path, 'output_opencv_sift_model')
    colmap.image_registrator(live_db_path, opencv_sift_base_model_path, opencv_sift_live_model_path)

    print('Live Model done!')

    print("Copying original live .db over..")
    shutil.copyfile(live_db_path, qt_db_path)  # overwrite live db with base db

    # 5 - replace the gt database's features with opencv sift features
    # gt db at this point is the same as live db
    gt_db = COLMAPDatabase.connect(qt_db_path)
    original_gt_db = COLMAPDatabase.connect(os.path.join(original_path, "gt", "database.db"))
    query_gt_images_txt_path = os.path.join(model_gt_path, "query_name.txt")
    image_names = np.loadtxt(query_gt_images_txt_path, dtype=str)  # only gt images

    print("Extracting data from images (gt only - ignoring the ones base and live as we already extracted data from them)..")

    for image_name in tqdm(image_names):
        # at this point we are looking at a gt image_id only.
        img = cv2.imread(os.path.join(images_gt_path, image_name))
        kps_plain = []
        kps, des = sift.detectAndCompute(img, None)
        if (des is None):  # this happens with textureless images, such as 71411677979.jpg
            continue
        # as in paper ~2000 for map, ~800 for query
        idxs = np.arange(len(kps))
        np.random.shuffle(idxs)
        # TODO: pick the same number of keypoints as in the original gt db
        rnd_idxs = idxs[0:query_features_limit]  # random idxs query one here
        kps = np.array(kps)
        kps = kps[rnd_idxs]  # replace with random
        des = np.array(des)
        des = des[rnd_idxs]  # replace with random
        dominant_orientations = countDominantOrientations(kps)

        kps_plain += [[kps[i].pt[0], kps[i].pt[1], kps[i].octave, kps[i].angle, kps[i].size, kps[i].response] for i in range(len(kps))]
        kps_plain = np.array(kps_plain)

        # copy from original gt db (note the fetching by image name and cam_id)
        img_details = original_gt_db.execute("SELECT * FROM images WHERE name = " + "'" + str(image_name) + "'").fetchone()
        gt_db.execute("INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (img_details[0],) + (img_details[1],) + (img_details[2],) + (img_details[3],) + (img_details[4],) + (img_details[5],) + (img_details[6],) + (
                        img_details[7],) + (img_details[8],) + (img_details[9],))
        cam_details = original_gt_db.execute("SELECT * FROM cameras WHERE camera_id = " + "'" + str(img_details[2]) + "'").fetchone()
        # INSERT OR IGNORE is used for CMU and Retail datasets, as they have base, live and gt images with the same three camera_ids
        gt_db.execute("INSERT OR IGNORE INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
                        (cam_details[0],) + (cam_details[1],) + (cam_details[2],) + (cam_details[3],) + (cam_details[4],) + (cam_details[5],))
        image_id = img_details[0]
        gt_db.insert_keypoints(image_id, kps_plain, dominant_orientations)
        gt_db.insert_descriptors(image_id, des)
        gt_db.commit()

    print("Matching -> at gt model")
    colmap.vocab_tree_matcher(qt_db_path, query_gt_images_txt_path)

    # 6 - register the new gt images against the base opencv sift model
    opencv_sift_gt_model_path = os.path.join(model_gt_path, 'output_opencv_sift_model')
    colmap.image_registrator(qt_db_path, opencv_sift_live_model_path, opencv_sift_gt_model_path)

    print('Gt Model done!')

# base_path here is for MnM
# This might help
# https://github.com/tsattler/visuallocalizationbenchmark/blob/master/local_feature_evaluation/modify_database_with_custom_features_and_matches.py
dataset = sys.argv[1] #HGE, CAB, LIN (or Other for CMU, retail shop)

if(dataset == "HGE" or dataset == "CAB" or dataset == "LIN"):
    doing_lamar = True
    base_path = f"/media/iNicosiaData/engd_data/lamar/{dataset}_colmap_model/models_for_match_no_match"
    remove_folder_safe(base_path)
    original_path = f"/media/iNicosiaData/engd_data/lamar/{dataset}_colmap_model"
    prepare_all_data_for_match_no_match(base_path, original_path, dataset, doing_lamar)

if(dataset == "CMU"):
    if(len(sys.argv) > 2):
        slices_names = [sys.argv[2]]
    else: # do all slices
        slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                        "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    for slice_name in slices_names:
        # overwrite paths
        base_path = f"/media/iNicosiaData/engd_data/cmu/{slice_name}/exmaps_data/models_for_match_no_match"
        remove_folder_safe(base_path)
        original_path = f"/media/iNicosiaData/engd_data/cmu/{slice_name}/exmaps_data"
        prepare_all_data_for_match_no_match(base_path, original_path, dataset, doing_lamar=False)

if(dataset == "RetailShop"):
    base_path = f"/media/iNicosiaData/engd_data/retail_shop/slice1/models_for_match_no_match"
    remove_folder_safe(base_path)
    original_path = f"/media/iNicosiaData/engd_data/retail_shop/slice1/"
    prepare_all_data_for_match_no_match(base_path, original_path, dataset, doing_lamar=False)


