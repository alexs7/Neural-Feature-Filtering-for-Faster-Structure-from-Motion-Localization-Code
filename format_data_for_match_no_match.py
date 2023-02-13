# Run this file before creating the data for MnM - (2020) paper, and run it on the CYENS machine.
# This file will aim to create the data for the RF model from Match no Match, MnM - (2020) paper
# It will extract OpenCV SIFT features and insert them in colmap's database and run the triangulator again for the base only.
# For the live and gt model, I just run the colmap's image_registrator, same as my first publication.
# The base model is triangulated using the already known camera poses from the original model
# I clear the old data, keypoints descriptors, and keep the poses (check COLMAP FAQ).
# You will need to run this on the CYENS machine as it has pycolmap and colmap installed - because of docker I can't run them on Bath Uni

# NOTE: 22/12/2022 The data can be kept on the CYENS machine. All the data is there now.

# Then get the 3D points averaged descriptors
# python3 get_points_3D_mean_desc_single_model_ml.py colmap_data/CMU_data/slice3_mnm/live/database.db colmap_data/CMU_data/slice3_mnm/live/output_opencv_sift_model/images.bin colmap_data/CMU_data/slice3_mnm/live/output_opencv_sift_model/points3D.bin colmap_data/CMU_data/slice3_mnm/avg_descs_xyz_ml.npy

# TODO: use the colmap exhaustive_matcher with --SiftMatching.use_gpu 1 for the query stage, if images are too many then no
# TODO: or train your own as here: https://github.com/colmap/colmap/issues/866

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
from helper import remove_folder_safe
from parameters import Parameters
from point3D_loader import read_points3d_default
from query_image import is_image_from_base, read_images_binary, get_image_name_from_db_with_id, get_all_images_names_from_db, get_image_id, \
    get_keypoints_data_and_dominantOrientations, get_descriptors

# The more you increase the values the more images will localise from live and gt
# The randomness is to simulate the COLMAP feature extraction 800 query / 2000 recon. used in previous papers
reconstr_features_limit = random.randint(1900, 2200)

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

def prepare_all_data_for_match_no_match(base_path, original_path, dataset=None, doing_lamar=False):
    # MnM model paths
    model_base_path = os.path.join(base_path, "base")
    model_live_path = os.path.join(base_path, "live")
    model_gt_path = os.path.join(base_path, "gt")

    # NOTE 13/02/2023:
    # Since I am running this on CYENS now I will copy base,live,gt models to the folder "base_path"
    print(f"Copying base, live, gt models to {base_path}..")
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
    manually_created_model_txt_path = os.path.join(model_base_path, 'empty_model_for_triangulation_txt')  # the "empty model" that will be used to create "opencv_sift_model"
    os.makedirs(manually_created_model_txt_path, exist_ok=True)

    # set up files as stated online in COLMAP's faq
    colmap_model_path = os.path.join(model_base_path, 'model')
    reconstruction = pycolmap.Reconstruction(colmap_model_path)
    # export model to txt
    print("Exporting model to txt..")
    reconstruction.write_text(manually_created_model_txt_path)
    points_3D_file_txt_path = os.path.join(manually_created_model_txt_path, 'points3D.txt')
    images_file_txt_path = os.path.join(manually_created_model_txt_path, 'images.txt')
    empty_points_3D_txt_file(points_3D_file_txt_path)
    arrange_images_txt_file(images_file_txt_path)
    base_db_path = os.path.join(model_base_path, 'database.db')

    base_db = COLMAPDatabase.connect(base_db_path)
    image_names = get_all_images_names_from_db(base_db)

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

    base_db.delete_all_matches()
    base_db.delete_all_two_view_geometries()
    base_db.commit()

    print("Matching -> at base model")
    colmap.vocab_tree_matcher(base_db_path)

    # 2 - triangulate the base model -> base opencv sift model
    opencv_sift_base_model_path = os.path.join(model_base_path, 'output_opencv_sift_model')
    colmap.point_triangulator(base_db_path, images_base_path, manually_created_model_txt_path, opencv_sift_base_model_path)

    print('Base Model done!')

    # 3 - replace the live database's features with opencv sift features (for all images, base + live)
    # live db also already contains the base images
    live_db_path = os.path.join(model_live_path, 'database.db')
    live_db = COLMAPDatabase.connect(live_db_path)
    image_names = get_all_images_names_from_db(live_db) #here are live images (+ base)

    if(live_db.dominant_orientations_column_exists() == False):
        live_db.add_dominant_orientations_column()
        live_db.commit() #we need to commit here

    print("Extracting data from images (live only - ignoring the ones base as we already extracted data from them)..")

    for image_name in tqdm(image_names):
        # we need to loop though all the live images here (including base too ofc)
        image_id = get_image_id(live_db, image_name)
        # check if image exists in base - which means its keypoints and descs have been already replaced, from previous iterations
        # image_id from live, will return None below, because we want it to, as at this point we
        # did not extract opencv data kps and des for that live image
        base_db_keypoints = get_keypoints_data_and_dominantOrientations(base_db, image_id)
        base_db_descriptors = get_descriptors(base_db, image_id)

        if((base_db_keypoints == None) or (base_db_descriptors == None)): #need to replace the kps and descs then in the live db
            # at this point we are looking at a live image_id only.
            # print(f'for image id = {image_id}, kps, des, and dominant_orientations has not be computed so will compute them and replace them in the live db')
            img = cv2.imread(os.path.join(images_live_path, image_name))
            kps_plain = []
            kps, des = sift.detectAndCompute(img,None)
            if (des is None):  # this happens with textureless images, such as 71411677979.jpg
                continue
            # as in paper ~2000 for map, ~800 for query
            idxs = np.arange(len(kps))
            np.random.shuffle(idxs)
            key_points_no = get_descriptors(live_db, image_id)[0] #same length as keypoints
            # pick the same number of keypoints as in the original live db
            rnd_idxs = idxs[0:key_points_no]  # random idxs query one here
            kps = np.array(kps)
            kps = kps[rnd_idxs]  # replace with random
            des = np.array(des)
            des = des[rnd_idxs]  # replace with random
            dominant_orientations = countDominantOrientations(kps)

            kps_plain += [[kps[i].pt[0], kps[i].pt[1], kps[i].octave, kps[i].angle, kps[i].size, kps[i].response] for i in range(len(kps))]
            kps_plain = np.array(kps_plain)

            live_db.replace_keypoints(image_id, kps_plain, dominant_orientations)
            live_db.replace_descriptors(image_id, des)
        else:
            # copy from base db the descriptors and keypoints already extracted to the live db
            # no need to re-compute dominant_orientations, sift kps and des
            # print(f'for image id = {image_id}, kps, des, and dominant_orientations has already computed so will load them from previous database, i.e. base')
            kps = base_db_keypoints[2]
            dominant_orientations = base_db_keypoints[3]
            kps_plain = []
            kps_plain += [[kps[i][0], kps[i][1], kps[i][2], kps[i][3], kps[i][4], kps[i][5]] for i in range(len(kps))]
            kps_plain = np.array(kps_plain)
            des = base_db_descriptors[2]
            live_db.replace_keypoints(image_id, kps_plain, dominant_orientations)
            live_db.replace_descriptors(image_id, des)

    live_db.delete_all_matches()
    live_db.delete_all_two_view_geometries()
    live_db.commit()

    print("Matching -> at live model")
    colmap.vocab_tree_matcher(live_db_path)

    # 4 - register the new live images against the base opencv sift model
    base_opencv_sift_model_path = os.path.join(model_base_path, 'output_opencv_sift_model')  #base model but with opencv sift features
    colmap.image_registrator(live_db_path,
                             base_opencv_sift_model_path,
                             os.path.join(model_live_path, 'output_opencv_sift_model'))

    print('Live Model done!')

    # 5 - replace the gt database's features with opencv sift features (for all images, base + live + gt)
    # gt db also already contains the base images + live images
    qt_db_path = os.path.join(model_gt_path, 'database.db')
    gt_db = COLMAPDatabase.connect(qt_db_path)
    image_names = get_all_images_names_from_db(gt_db) #here are gt images (+ base + live)

    if(gt_db.dominant_orientations_column_exists() == False):
        gt_db.add_dominant_orientations_column() #np.float32 is stored in here
        gt_db.commit() #we need to commit here

    print("Extracting data from images (gt only - ignoring the ones from live and base as we already extracted data from them)..")

    for image_name in tqdm(image_names):
        # we need to loop though all the gt images here (including live and base ofc)
        image_id = get_image_id(gt_db, image_name)
        # check if image exists in live - which means its keypoints and descs have been already replaced, from previous iterations
        # image_id from gt, will return None below, because we want it to, as at this point we
        # did not extract opencv data kps and des for that gt image
        live_db_keypoints = get_keypoints_data_and_dominantOrientations(live_db, image_id)
        live_db_descriptors = get_descriptors(live_db, image_id)

        if((live_db_keypoints == None) or (live_db_descriptors == None)): #need to replace the kps and descs then
            # at this point we are looking at a gt image_id only.
            # print(f'for image id = {image_id}, kps, des, and dominant_orientations has not be computed so will computer them and replace them in the gt db')
            img = cv2.imread(os.path.join(images_gt_path, image_name))
            kps_plain = []
            # here we keep all of the query descriptors.
            # This is because these descriptors will be used for training. The more the better.
            kps, des = sift.detectAndCompute(img, None)
            if (des is None):  # this happens with textureless images, such as 71411677979.jpg
                continue
            # as in paper ~2000 for map, ~800 for query
            idxs = np.arange(len(kps))
            np.random.shuffle(idxs)
            key_points_no = get_descriptors(gt_db, image_id)[0]  # same length as keypoints, at this point the image is in the gt db
            # pick the same number of keypoints as in the original gt db
            rnd_idxs = idxs[0:key_points_no]  # random idxs query one here
            kps = np.array(kps)
            kps = kps[rnd_idxs]  # replace with random
            des = np.array(des)
            des = des[rnd_idxs]  # replace with random
            dominant_orientations = countDominantOrientations(kps)

            kps_plain += [[kps[i].pt[0], kps[i].pt[1], kps[i].octave, kps[i].angle, kps[i].size, kps[i].response] for i in range(len(kps))]
            kps_plain = np.array(kps_plain)

            gt_db.replace_keypoints(image_id, kps_plain, dominant_orientations)
            gt_db.replace_descriptors(image_id, des)
        else:
            # copy from live db the descriptors and keypoints already extracted to the gt db
            # no need to re-compute dominant_orientations, sift kps and des
            # print(f'for image id = {image_id}, kps, des, and dominant_orientations has already computed so will load them from previous database, i.e. live')
            kps = live_db_keypoints[2]
            dominant_orientations = live_db_keypoints[3]
            kps_plain = []
            kps_plain += [[kps[i][0], kps[i][1], kps[i][2], kps[i][3], kps[i][4], kps[i][5]] for i in range(len(kps))]
            kps_plain = np.array(kps_plain)
            des = live_db_descriptors[2]
            gt_db.replace_keypoints(image_id, kps_plain, dominant_orientations)
            gt_db.replace_descriptors(image_id, des)

    gt_db.delete_all_matches()
    gt_db.delete_all_two_view_geometries()
    gt_db.commit()

    print("Matching -> at gt model")
    colmap.vocab_tree_matcher(qt_db_path)

    # 6 - register the new gt images against the live opencv sift model
    live_opencv_sift_model_path = os.path.join(model_live_path, 'output_opencv_sift_model')  #live model but with opencv sift features
    colmap.image_registrator(qt_db_path,
                             live_opencv_sift_model_path,
                             os.path.join(model_gt_path, 'output_opencv_sift_model'))

    print('Gt Model done!')

# base_path here is for MnM

dataset = sys.argv[1] #HGE, CAB, LIN (or Other for CMU, retail shop)

if(dataset == "HGE" or dataset == "CAB" or dataset == "LIN"):
    doing_lamar = True
    base_path = f"/media/iNicosiaData/engd_data/lamar/{dataset}_colmap_model/models_for_match_no_match"
    remove_folder_safe(base_path)
    original_path = f"/media/iNicosiaData/engd_data/lamar/{dataset}_colmap_model"
    prepare_all_data_for_match_no_match(base_path, original_path, dataset, doing_lamar)

if(dataset == "CMU"):
    slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                    "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    for slice_name in slices_names:
        # overwrite paths
        base_path = f"/media/iNicosiaData/engd_data/cmu/{slice_name}/exmaps_data/models_for_match_no_match"
        # remove_folder_safe(base_path)
        original_path = f"/media/iNicosiaData/engd_data/cmu/{slice_name}/exmaps_data"
        prepare_all_data_for_match_no_match(base_path, original_path, dataset, doing_lamar=False)

if(dataset == "RetailShop"):
    base_path = f"/media/iNicosiaData/engd_data/retail_shop/slice1/models_for_match_no_match"
    remove_folder_safe(base_path)
    original_path = f"/media/iNicosiaData/engd_data/retail_shop/slice1/"
    prepare_all_data_for_match_no_match(base_path, original_path, dataset, doing_lamar=False)


