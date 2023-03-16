# Run this file before creating the data for MnM - (2020) paper, and run it on the CYENS machine.
# This file will create OpenCV models for each dataset, so the comparison are fair.
# It will extract OpenCV SIFT features and insert them in colmap's database, run custom mapper using original's COLMAP models pairs from Exmaps,
# and run the triangulator again to create the gt model only.
# The gt model can be used to apply exp decay if needed (just restrict the images to NOT include the gt ones, hence live map).
# You will need to run this on the CYENS machine as it has pycolmap and colmap installed - because of docker I can't run them on Bath Uni
# This file will use the image pairs from ExMaps models to avoid using vocabulary tree for image retrieval.
# The SIFT matching still happens here (in the custom_matcher).
# Don't forget to run get_points_3D_mean_desc_ml_mnm.py after this file to get the avg of the 3D desc of a point same order as in points3D

import os
import random
import shutil
import sys
import cv2
import numpy as np
import pycolmap
import sklearn
from sklearn import preprocessing
from scantools.utils.colmap import read_images_binary, read_points3D_binary
from tqdm import tqdm
import colmap
from database import COLMAPDatabase, pair_id_to_image_ids
from helper import remove_folder_safe
from query_image import get_all_images_names_from_db, get_image_id, get_localised_image_by_names, get_full_image_name_from_db_with_id, get_descriptors
from save_2D_points import save_debug_image_simple_ml
import torch #installed with conda install pytorch torchvision torchaudio pytorch-cuda=11.7 cudatoolkit=11.2.2  -c pytorch -c nvidia (on CYENS machine)

# NOTE: if you get an error about symbol cublasLtGetStatusString version libcublasLt.so.11, then run this:
# export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if(use_cuda):
    print("PyTorch, using CUDA")

# https://github.com/tsattler/visuallocalizationbenchmark/blob/1f8e4311e6fc41519e3d9909e14fce9a7f013732/local_feature_evaluation/matchers.py
# Mutual NN + symmetric Lowe's ratio test matcher for L2 normalized descriptors.
def mutual_nn_ratio_matcher(descriptors1, descriptors2, ratio=0.8):
    device = descriptors1.device
    # similarity matrix
    sim = descriptors1 @ descriptors2.t()

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN and match similarity.
    nn12 = nns[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]

    # Mutual NN + symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ids1 == nn21[nn12], torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio))

    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)

    return matches.data.cpu().numpy()

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

def prepare_all_data_for_match_no_match(mnm_path, original_path, dataset=None, doing_lamar=False, nfeatures=-1):
    if(nfeatures == -1):
        raise Exception("nfeatures must be set to a value")
    # MnM model paths
    model_gt_path = os.path.join(mnm_path, "gt")

    # remove old models and their data
    print(f"Removing previous MnM gt data folder, i.e. {model_gt_path}..")
    remove_folder_safe(model_gt_path)

    # NOTE 13/02/2023:
    # Since I am running this on CYENS now I will copy base to the folder "base_path"
    # only need the gt model here
    print(f"Copying gt..")
    shutil.copytree(os.path.join(original_path, "gt"), model_gt_path, dirs_exist_ok=True)

    # MnM database paths
    qt_db_path = os.path.join(model_gt_path, 'database.db')
    gt_db = COLMAPDatabase.connect(qt_db_path)  # MnM database can modify it

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

    # look at cmu_sparse_reconstuctor.py, for help
    # Note: use images names from database to locate them for opencv feature extraction

    manually_created_model_txt_path = os.path.join(model_gt_path, 'empty_model_for_triangulation_txt')  # the "empty model" that will be used to create "opencv_sift_model"
    os.makedirs(manually_created_model_txt_path, exist_ok=True)

    opencv_points_images_path = os.path.join(model_gt_path, 'opencv_points_images')  # to see openCV's points
    os.makedirs(opencv_points_images_path, exist_ok=True)

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

    # when running for the first time it is OK to add columns
    print("Adding columns to database..")
    gt_db.add_octaves_column()
    gt_db.add_angles_column()
    gt_db.add_sizes_column()
    gt_db.add_responses_column()
    gt_db.add_green_intensities_column()
    gt_db.add_dominant_orientations_column()
    gt_db.add_matched_column()
    gt_db.add_images_localised_column()
    gt_db.add_images_is_base_column()
    gt_db.add_images_is_live_column()
    gt_db.add_images_is_gt_column()
    gt_db.commit()

    image_names = get_all_images_names_from_db(gt_db)  # base + live + gt

    print("Moving images so it is easier to find them..")
    for image_name in tqdm(image_names):
        # base image
        if os.path.exists(os.path.join(images_base_path, image_name)):
            image_file_path = os.path.join(images_base_path, image_name)
            copy_image_to_all_images_folder(image_file_path, all_images_path, image_name)
            gt_db.update_images_is_base_value(image_name)
        # live image
        if os.path.exists(os.path.join(images_live_path, image_name)):
            image_file_path = os.path.join(images_live_path, image_name)
            copy_image_to_all_images_folder(image_file_path, all_images_path, image_name)
            gt_db.update_images_is_live_value(image_name)
        # gt image
        if os.path.exists(os.path.join(images_gt_path, image_name)):
            image_file_path = os.path.join(images_gt_path, image_name)
            copy_image_to_all_images_folder(image_file_path, all_images_path, image_name)
            gt_db.update_images_is_gt_value(image_name)

    gt_db.commit()
    print("At this point you can start training the C++ MnM RF model..")

    print("Extracting features from images and inserting in db (gt)..")
    for image_name in tqdm(image_names):
        # base image
        if os.path.exists(os.path.join(images_base_path, image_name)):
            image_file_path = os.path.join(images_base_path, image_name)
        # live image
        if os.path.exists(os.path.join(images_live_path, image_name)):
            image_file_path = os.path.join(images_live_path, image_name)
        # gt image
        if os.path.exists(os.path.join(images_gt_path, image_name)):
            image_file_path = os.path.join(images_gt_path, image_name)

        img = cv2.imread(image_file_path)
        image_id = get_image_id(gt_db, image_name)

        # COLMAP will always detect more SIFT than OpenCV.
        # For example if you set SIFT_create() features to the number of keypoints in the original COLMAP db,
        # A number of images will have more COLMAP SIFT features than OpenCV SIFT features.
        sift = cv2.SIFT_create(nfeatures=nfeatures)  # NOTE: You can set the max feature parameter here to match the number of keypoints in the original model
        kps, des = sift.detectAndCompute(img, None)

        if(des is None): #in theis case kps is ()
            # can't get descriptors for this image, so remove its keypoints, descriptors from the database
            # or easy way is to just continue
            gt_db.replace_keypoints(image_id, np.zeros([0,2]))
            gt_db.replace_descriptors(image_id, np.zeros([0,128]))
            gt_db.commit()
            print("No descriptors for image (None): " + image_name)
            continue

        if(des.shape[0] == 0 or des.shape[0] == 1):
            # because sometimes it returns an empty array or only 1 which is not enough for matching later on (k=2)
            # insert zero no of keypoints and descriptors
            gt_db.replace_keypoints(image_id, np.zeros([0, 2]))
            gt_db.replace_descriptors(image_id, np.zeros([0, 128]))
            gt_db.commit()
            print("No descriptors for image (len == 0): " + image_name)
            continue

        # pick the same number of keypoints as the original database - not needed just kept for reference
        # gt_db_original_kps_no = gt_db_original.execute("SELECT rows FROM keypoints WHERE image_id = ?", (image_id,)).fetchone()[0]

        kps = np.array(kps)
        des = np.array(des)

        kps_plain = [[kps[i].pt[0], kps[i].pt[1]] for i in range(len(kps))]
        assert len(kps_plain) == len(des)
        assert len(kps_plain) == len(kps)
        kps_plain = np.array(kps_plain)

        # just visualising the openCV keypoints
        save_debug_image_simple_ml(image_file_path, kps_plain, kps_plain, os.path.join(opencv_points_images_path, f"{image_id}.jpg"))

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
    # at this point gt_db has the same matches as gt_db_original
    all_pairs = gt_db.execute("SELECT pair_id FROM matches WHERE rows > 0").fetchall() #I do not need pairs with no matches
    f = open(pairs_txt_path, 'a')
    for pair_id in tqdm(all_pairs):
        img_1_id, img_2_id = pair_id_to_image_ids(pair_id[0])
        img_1_name = get_full_image_name_from_db_with_id(gt_db, img_1_id)
        img_2_name = get_full_image_name_from_db_with_id(gt_db, img_2_id)
        pair = [f"{img_1_name} {img_2_name}"]
        np.savetxt(f, pair, fmt="%s")
    f.close()

    # At this point clear the matches
    print("Deleting previous matches..")
    gt_db.delete_all_matches()

    print("Doing my own feature matching..")  # for custom matcher too
    pairs_with_matches = {}
    for pair_id in tqdm(all_pairs):
        pair_id = pair_id[0]
        img_1_id, img_2_id = pair_id_to_image_ids(pair_id)
        # names kept but not used
        img_1_name = get_full_image_name_from_db_with_id(gt_db, img_1_id)
        img_2_name = get_full_image_name_from_db_with_id(gt_db, img_2_id)
        # note the -1 at the end, because the last element of the list is the descriptors
        img_1_descs = get_descriptors(gt_db, str(img_1_id))[-1]
        img_2_descs = get_descriptors(gt_db, str(img_2_id))[-1]
        if(img_1_descs.shape[0] == 0 or img_2_descs.shape[0] == 0):
            print("No descriptors for one of the images in pair: " + str(pair_id))
            continue #skip this pair as there are not descs for one of the images
        img_1_descs = sklearn.preprocessing.normalize(img_1_descs, norm='l2')  # as suggested in https://github.com/colmap/colmap/issues/578
        img_2_descs = sklearn.preprocessing.normalize(img_2_descs, norm='l2')
        descriptors1 = torch.from_numpy(img_1_descs).to(device)
        descriptors2 = torch.from_numpy(img_2_descs).to(device)
        matches = mutual_nn_ratio_matcher(descriptors1, descriptors2).astype(np.uint32)
        if img_1_id > img_2_id:
            matches = matches[:, [1, 0]]
        pairs_with_matches[pair_id] = matches
        # gt_db.insert_matches(pair_id, matches.shape[0], matches.shape[1], matches) #uncomment if you use "pairs" in custom_matcher, do not use if you use "raw"

    gt_db.delete_all_two_view_geometries()
    gt_db.commit()

    print("Writing pairs /w matches to .txt file..")  # for custom matcher
    pairs_with_matches_txt_path = os.path.join(model_gt_path, 'pairs_with_matches.txt')
    # at this point gt_db has the same matches as gt_db_original
    f = open(pairs_with_matches_txt_path, 'a')
    for pair_id, pair_matches in tqdm(pairs_with_matches.items()):
        img_1_id, img_2_id = pair_id_to_image_ids(pair_id)
        img_1_name = get_full_image_name_from_db_with_id(gt_db, img_1_id)
        img_2_name = get_full_image_name_from_db_with_id(gt_db, img_2_id)
        pair = [f"{img_1_name} {img_2_name}"]

        np.savetxt(f, pair, fmt="%s")
        np.savetxt(f, pair_matches, fmt="%d")
        f.write("\n")
    f.close()

    print("Geometry verification..")
    # This is the old one that uses only pairs and does the matching again

    # NOTE: You can insert your own matches and run custom_matcher with "pairs" - not recommended
    # The proper way since you are doing the matching is to use "raw" and use pairs_with_matches_txt_path
    # The code in this file will do the latter, although I generated the data using the first method.
    # I compared the matched and two_view_geometries table and points cloud and they are the almost identical.

    # colmap.custom_matcher(qt_db_path, pairs_txt_path, match_type = "pairs")
    colmap.custom_matcher(qt_db_path, pairs_with_matches_txt_path, match_type = "raw") #this will not run matching again

    opencv_sift_gt_model_path = os.path.join(model_gt_path, 'output_opencv_sift_model')
    # Triangulate points, and load images from all_images_path so it is easier with different images' name
    colmap.point_triangulator(qt_db_path, all_images_path, manually_created_model_txt_path, opencv_sift_gt_model_path)

    print("Inserting matched values to gt db, from localised images..")
    gt_images = read_images_binary(os.path.join(opencv_sift_gt_model_path, "images.bin"))
    gt_images_names = [image.name for id, image in gt_images.items()] #localised
    print("Loading 3D points..")
    gt_points3D = read_points3D_binary(os.path.join(opencv_sift_gt_model_path, "points3D.bin"))
    # Note: all localised_qt_images_names will have matched values, i.e.  0 or 1 for each keypoint but not 99 for all keypoints (default value)

    for image_name in tqdm(image_names):
        img_id = int(get_image_id(gt_db, image_name))
        keypoints_no_db = gt_db.execute("SELECT rows FROM keypoints WHERE image_id = ?", (img_id,)).fetchone()[0]
        matched_values = []

        if(image_name not in gt_images_names):
            for i in range(keypoints_no_db):
                matched_values.append(0)
            gt_db.update_matched_values(img_id, matched_values)
            continue #then it is not localised value will be 0 by default, matches will be [0,0,0,...] as well

        image = gt_images[img_id]
        gt_db.update_images_localised_value(img_id, 1) #set image to localised in db, easier to find later
        db_name = gt_db.execute("SELECT name FROM images WHERE image_id = ?", (img_id,)).fetchone()[0]
        # sanity checks
        assert db_name == image_name

        if(image.xys.shape[0] == 0): #for the rare case that image has no triangulated points but keypoints in db
            for i in range(keypoints_no_db):
                matched_values.append(0)
            gt_db.update_matched_values(img_id, matched_values)
        else:
            for i in range(image.xys.shape[0]):
                assert keypoints_no_db == image.xys.shape[0] #sanity check
                current_point3D_id = image.point3D_ids[i]
                if (current_point3D_id == -1):  # means feature is unmatched
                    matched = 0
                else:
                    # this is to make sure that xy belong to the right pointd3D
                    assert i in gt_points3D[current_point3D_id].point2D_idxs
                    matched = 1
                matched_values.append(matched)
            gt_db.update_matched_values(img_id, matched_values)

    # at this point no image should have a value of 99 in the matched column in the keypoints table
    gt_db.commit()
    print('Done!')

# base_path here is for MnM
# This might help
# https://github.com/tsattler/visuallocalizationbenchmark/blob/master/local_feature_evaluation/modify_database_with_custom_features_and_matches.py
dataset = sys.argv[1] #HGE, CAB, LIN (or Other for CMU, retail shop)
nfeatures = int(sys.argv[2])

if(dataset == "HGE" or dataset == "CAB" or dataset == "LIN"):
    mnm_path = f"/media/iNicosiaData/engd_data/lamar/{dataset}_colmap_model/models_for_match_no_match"
    # remove_folder_safe(base_path)
    original_path = f"/media/iNicosiaData/engd_data/lamar/{dataset}_colmap_model"
    prepare_all_data_for_match_no_match(mnm_path, original_path, dataset, doing_lamar=True, nfeatures=nfeatures)

if(dataset == "CMU"):
    if(len(sys.argv) > 3):
        slices_names = [sys.argv[3]]
    else: # do all slices
        slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                    "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    for slice_name in slices_names:
        # overwrite paths
        mnm_path = f"/media/iNicosiaData/engd_data/cmu/{slice_name}/exmaps_data/models_for_match_no_match"
        original_path = f"/media/iNicosiaData/engd_data/cmu/{slice_name}/exmaps_data"
        prepare_all_data_for_match_no_match(mnm_path, original_path, dataset, doing_lamar=False, nfeatures=nfeatures)

if(dataset == "RetailShop"):
    mnm_path = f"/media/iNicosiaData/engd_data/retail_shop/slice1/models_for_match_no_match"
    # remove_folder_safe(base_path)
    original_path = f"/media/iNicosiaData/engd_data/retail_shop/slice1/"
    prepare_all_data_for_match_no_match(mnm_path, original_path, dataset, doing_lamar=False, nfeatures=nfeatures)

