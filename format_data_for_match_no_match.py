# Run this file before creating the data for MnM - (2020) paper, and run it on the CYENS machine.
# This file will aim to create the data for the RF model from Match no Match, MnM - (2020) paper
# It will extract OpenCV SIFT features and insert them in colmap's database and run the triangulator again for the gt only.
# You will need to run this on the CYENS machine as it has pycolmap and colmap installed - because of docker I can't run them on Bath Uni
# This file will use the image pairs from ExMaps models to avoid using vocabulary tree for image retrieval.
# The SIFT matching still happens here (in custom_matcher)

import os
import random
import shutil
import sys
import cv2
import numpy as np
import pycolmap
from scantools.utils.colmap import read_images_binary, read_points3D_binary
from tqdm import tqdm
import colmap
from database import COLMAPDatabase, pair_id_to_image_ids
from helper import remove_folder_safe
from query_image import get_all_images_names_from_db, get_image_id, get_localised_image_by_names, get_full_image_name_from_db_with_id

def countDominantOrientations(keypoints): #MnM code
    domOrientations = np.ones([len(keypoints), 1])
    # notice I get two xs and two ys from the same source, so I can compare them
    x1 = np.array([kp[0] for kp in keypoints])
    x2 = np.array([kp[0] for kp in keypoints])
    y1 = np.array([kp[1] for kp in keypoints])
    y2 = np.array([kp[1] for kp in keypoints])

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

def copy_image_to_all_images_folder(image_file_path, all_images_path, image_name):
    os.makedirs(os.path.dirname(os.path.join(all_images_path, image_name)), exist_ok=True)
    shutil.copyfile(image_file_path, os.path.join(all_images_path, image_name))
    pass

def prepare_all_data_for_match_no_match(mnm_path, original_path, dataset=None, doing_lamar=False):
    # MnM model paths
    model_gt_path = os.path.join(mnm_path, "gt")
    # MnM database paths
    qt_db_path = os.path.join(model_gt_path, 'database.db')

    # remove old models and their data
    print(f"Removing old MnM models and their data, i.e. {mnm_path}..")
    remove_folder_safe(mnm_path)

    # NOTE 13/02/2023:
    # Since I am running this on CYENS now I will copy base to the folder "base_path"
    # only need the gt model here
    print(f"Copying gt..")
    shutil.copytree(os.path.join(original_path, "gt"), model_gt_path, dirs_exist_ok=True)
    # I will need the original database for getting the number of keypoints
    qt_db_original_path = os.path.join(original_path, "gt", 'database.db')
    gt_db_original = COLMAPDatabase.connect(qt_db_original_path)

    # use original path here as that is where the original images are
    if(doing_lamar == True):
        images_base_path = os.path.join("/media/iNicosiaData/engd_data/lamar/", dataset, "sessions", "map", "raw_data")
        images_live_path = os.path.join("/media/iNicosiaData/engd_data/lamar/", dataset, "sessions", "query_phone", "raw_data")
        images_gt_path = os.path.join("/media/iNicosiaData/engd_data/lamar/", dataset, "sessions", "query_val_phone", "raw_data")
    else:
        images_base_path = os.path.join(os.path.join(original_path, "base"), "images")
        images_live_path = os.path.join(os.path.join(original_path, "live"), "images")
        images_gt_path = os.path.join(os.path.join(original_path, "gt"), "images")

    print("Removing old images..")
    all_images_path = os.path.join(mnm_path, "all_images")
    remove_folder_safe(all_images_path)

    sift = cv2.SIFT_create()

    # look at cmu_sparse_reconstuctor.py, for help
    # Note: use images names from database to locate them for opencv feature extraction

    manually_created_model_txt_path = os.path.join(model_gt_path, 'empty_model_for_triangulation_txt')  # the "empty model" that will be used to create "opencv_sift_model"
    os.makedirs(manually_created_model_txt_path, exist_ok=True)

    # The original model is from ExMaps, get_lamar.py or get_cmu etc. It has been used to run benchmarks, so it is reliable enough to use (it was copied over).
    colmap_model_path = os.path.join(model_gt_path, 'model')  # the original model is in here, since it was copied over from ExMaps
    reconstruction = pycolmap.Reconstruction(colmap_model_path)  # loading the original model from Exmaps

    # set up files as stated online in COLMAP's faq
    # export model to txt
    print("Exporting model to txt..")
    reconstruction.write_text(manually_created_model_txt_path)  # this will create the files: cameras.txt, images.txt, points3D.txt
    points_3D_file_txt_path = os.path.join(manually_created_model_txt_path, 'points3D.txt')
    images_file_txt_path = os.path.join(manually_created_model_txt_path, 'images.txt')
    empty_points_3D_txt_file(points_3D_file_txt_path)  # as in COLMAP's faq
    arrange_images_txt_file(images_file_txt_path)  # as in COLMAP's faq

    gt_db = COLMAPDatabase.connect(qt_db_path) #MnM database can modify it

    # when running for the first time it is OK to add columns
    gt_db.add_octaves_column()
    gt_db.add_angles_column()
    gt_db.add_sizes_column()
    gt_db.add_responses_column()
    gt_db.add_green_intensities_column()
    gt_db.add_dominant_orientations_column()
    gt_db.add_matched_column()

    image_names = get_all_images_names_from_db(gt_db)  # base + live + gt

    print("Moving images so it is easier to find them..")
    for image_name in tqdm(image_names):
        # base image
        if os.path.exists(os.path.join(images_base_path, image_name)):
            image_file_path = os.path.join(images_base_path, image_name)
            copy_image_to_all_images_folder(image_file_path, all_images_path, image_name)
        # live image
        if os.path.exists(os.path.join(images_live_path, image_name)):
            image_file_path = os.path.join(images_live_path, image_name)
            copy_image_to_all_images_folder(image_file_path, all_images_path, image_name)
        # gt image
        if os.path.exists(os.path.join(images_gt_path, image_name)):
            image_file_path = os.path.join(images_gt_path, image_name)
            copy_image_to_all_images_folder(image_file_path, all_images_path, image_name)

    print("At this point you can start training the C++ MnM RF model..")

    print("Extracting features from images and inserting in db (gt)..")
    for image_name in tqdm(image_names):
        # base image
        if os.path.exists(os.path.join(images_base_path, image_name)):
            image_file_path = os.path.join(images_base_path, image_name)
            copy_image_to_all_images_folder(image_file_path, all_images_path, image_name)
        # live image
        if os.path.exists(os.path.join(images_live_path, image_name)):
            image_file_path = os.path.join(images_live_path, image_name)
            copy_image_to_all_images_folder(image_file_path, all_images_path, image_name)
        # gt image
        if os.path.exists(os.path.join(images_gt_path, image_name)):
            image_file_path = os.path.join(images_gt_path, image_name)
            copy_image_to_all_images_folder(image_file_path, all_images_path, image_name)

        img = cv2.imread(image_file_path)
        image_id = get_image_id(gt_db, image_name)

        kps, des = sift.detectAndCompute(img, None)
        if(des is None):
            # can't get descriptors for this image, so remove its keypoints, descriptors from the database
            # or easy way is to just continue
            continue
        if(des.shape[0] == 0):
            # because sometimes it returns an empty array
            continue

        idxs = np.arange(len(kps))
        # choose random keypoints
        np.random.shuffle(idxs)
        # pick the same number of keypoints as the original database
        gt_db_original_kps_no = gt_db_original.execute("SELECT rows FROM keypoints WHERE image_id = ?", (image_id,)).fetchone()[0]

        if(gt_db_original_kps_no == 0):
            # no keypoints in the original database, so continue
            continue

        rnd_idxs = idxs[0: gt_db_original_kps_no]
        kps = np.array(kps)
        kps = kps[rnd_idxs]  # replace with random
        des = np.array(des)
        des = des[rnd_idxs]  # replace with random

        kps_plain = [[kps[i].pt[0], kps[i].pt[1]] for i in range(len(kps))]
        assert len(kps_plain) == len(des)
        assert len(kps_plain) == len(kps)
        kps_plain = np.array(kps_plain)

        # run replace_keypoints and replace_descriptors first
        gt_db.replace_keypoints(image_id, kps_plain)
        gt_db.replace_descriptors(image_id, des)
        # then, insert additional MnM info
        gt_db.update_octaves(image_id, np.array([kps[i].octave for i in range(len(kps))]))
        gt_db.update_angles(image_id, np.array([kps[i].angle for i in range(len(kps))]))
        gt_db.update_sizes(image_id, np.array([kps[i].size for i in range(len(kps))]))
        gt_db.update_responses(image_id, np.array([kps[i].response for i in range(len(kps))]))
        green_intensities = []
        for i in range(kps_plain.shape[0]):
            x = kps_plain[i, 0]
            y = kps_plain[i, 1]
            green_intensity = img[int(y), int(x)][1]  # reverse indexing
            green_intensities.append(green_intensity)
        gt_db.update_green_intensities(image_id, green_intensities)
        dominant_orientations = countDominantOrientations(kps_plain)
        gt_db.update_dominant_orientations(image_id, dominant_orientations)
        gt_db.commit()

    print("Writing pairs to .txt file..") #for custom matcher
    pairs_txt_path = os.path.join(model_gt_path, 'pairs.txt')
    all_pairs = gt_db.execute("SELECT pair_id FROM matches WHERE rows > 0").fetchall() #I do not need pairs with no matches
    f = open(pairs_txt_path, 'a')
    for pair_id in tqdm(all_pairs):
        img_1_id, img_2_id = pair_id_to_image_ids(pair_id[0])
        img_1_name = get_full_image_name_from_db_with_id(gt_db, img_1_id)
        img_2_name = get_full_image_name_from_db_with_id(gt_db, img_2_id)
        pair = [f"{img_1_name} {img_2_name}"]
        np.savetxt(f, pair, fmt="%s")
    f.close()

    print("Deleting previous matches..")
    gt_db.delete_all_matches()
    gt_db.delete_all_two_view_geometries()
    gt_db.commit()

    print("Custom matching -> at gt model")
    colmap.custom_matcher(qt_db_path, pairs_txt_path, match_type = "pairs")

    opencv_sift_gt_model_path = os.path.join(model_gt_path, 'output_opencv_sift_model')
    # Triangulate points, and load images from all_images_path so it is easier with different images' name
    colmap.point_triangulator(qt_db_path, all_images_path, manually_created_model_txt_path, opencv_sift_gt_model_path)

    # get gt localised images number
    query_gt_images_txt_path = os.path.join(model_gt_path, "query_name.txt")
    gt_image_names = np.loadtxt(query_gt_images_txt_path, dtype=str)  # only gt images
    localised_qt_images_names = get_localised_image_by_names(gt_image_names, os.path.join(opencv_sift_gt_model_path, "images.bin"))  # only gt images (localised only)
    print(f"Total number of gt localised images: {len(localised_qt_images_names)}")
    print(f"Total number of gt (query) images (.txt): {len(gt_image_names)}")
    np.savetxt(os.path.join(mnm_path, "localised_qt_images_names.txt"), np.array([len(localised_qt_images_names)]))

    print("Inserting matched values to gt db, from localised images..")
    gt_images = read_images_binary(os.path.join(opencv_sift_gt_model_path, "images.bin"))
    gt_points3D = read_points3D_binary(os.path.join(opencv_sift_gt_model_path, "points3D.bin"))
    no_kps = 0
    more_kps = 0
    for image_name in tqdm(localised_qt_images_names):
        matched_values = []
        img_id = int(get_image_id(gt_db, image_name))
        image = gt_images[img_id]
        keypoints_no_db = gt_db.execute("SELECT rows FROM keypoints WHERE image_id = ?", (img_id,)).fetchone()[0]
        keypoints_no_original_db = gt_db_original.execute("SELECT rows FROM keypoints WHERE image_id = ?", (img_id,)).fetchone()[0]

        if(image.xys.shape[0] == 0):
            no_kps +=1
            continue
        if(keypoints_no_original_db > keypoints_no_db):
            more_kps +=1
            continue

        assert keypoints_no_db == keypoints_no_original_db
        assert keypoints_no_db == image.xys.shape[0]
        for i in range(image.xys.shape[0]):  # can loop through descs or img_data.xys - same thing
            current_point3D_id = image.point3D_ids[i]
            if (current_point3D_id == -1):  # means feature is unmatched
                matched = 0
            else:
                # this is to make sure that xy belong to the right pointd3D
                assert i in gt_points3D[current_point3D_id].point2D_idxs
                matched = 1
            matched_values.append(matched)
        gt_db.update_matched_values(img_id, matched_values)

    gt_db.commit()
    print(f"{no_kps} images had no keypoints and {more_kps} had more keypoints than the original ones")
    print('Done!')

# base_path here is for MnM
# This might help
# https://github.com/tsattler/visuallocalizationbenchmark/blob/master/local_feature_evaluation/modify_database_with_custom_features_and_matches.py
dataset = sys.argv[1] #HGE, CAB, LIN (or Other for CMU, retail shop)

if(dataset == "HGE" or dataset == "CAB" or dataset == "LIN"):
    doing_lamar = True
    mnm_path = f"/media/iNicosiaData/engd_data/lamar/{dataset}_colmap_model/models_for_match_no_match"
    # remove_folder_safe(base_path)
    original_path = f"/media/iNicosiaData/engd_data/lamar/{dataset}_colmap_model"
    prepare_all_data_for_match_no_match(mnm_path, original_path, dataset, doing_lamar)

if(dataset == "CMU"):
    if(len(sys.argv) > 2):
        slices_names = [sys.argv[2]]
    else: # do all slices
        slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                        "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    for slice_name in slices_names:
        # overwrite paths
        mnm_path = f"/media/iNicosiaData/engd_data/cmu/{slice_name}/exmaps_data/models_for_match_no_match"
        original_path = f"/media/iNicosiaData/engd_data/cmu/{slice_name}/exmaps_data"
        prepare_all_data_for_match_no_match(mnm_path, original_path, dataset, doing_lamar=False)

if(dataset == "RetailShop"):
    mnm_path = f"/media/iNicosiaData/engd_data/retail_shop/slice1/models_for_match_no_match"
    # remove_folder_safe(base_path)
    original_path = f"/media/iNicosiaData/engd_data/retail_shop/slice1/"
    prepare_all_data_for_match_no_match(mnm_path, original_path, dataset, doing_lamar=False)


