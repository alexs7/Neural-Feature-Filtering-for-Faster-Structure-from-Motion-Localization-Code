# Run this file before creating the data for MnM - (2020) paper
# This file will aim to create the data for the RF model from Match no Match, MnM - (2020) paper
# It will extract OpenCV features and insert them in colmap's database and run the triangulator again for the base only.
# For the live and gt model, I just run the image _registrator.
# The base model is triangulated using the already known camera poses from the original model
# I clear the old data, keypoints descriptors, and keep the poses (check COLMAP FAQ).
# You will need to run this on the CYENS machine as it has pycolmap and colmap installed - because of docker I can't run them on Bath Uni
# To start we need to copy base,live,gt to CYENS then run this script for each base,live,gt ; (scp -r -P 11568 base live gt  alex@4.tcp.eu.ngrok.io:/home/alex/uni/models_for_match_no_match/CMU_slice_3/)
# When the script is done then move the generated files back to bath servers, scp -r CMU_slice_3 ar2056@weatherwax.cs.bath.ac.uk:/mnt/fast1/ar2056/mnm_match_data
# Due to user ownerships I have to move them to ar2056@weatherwax.cs.bath.ac.uk:/mnt/fast1/ar2056/ then to the approriate folder.

# Once done copy back to the appropriate CMU or Coop data folder for MnM in weatherwax and run:
# python3 get_points_3D_mean_desc_single_model_ml.py colmap_data/CMU_data/slice3_mnm/live/database.db colmap_data/CMU_data/slice3_mnm/live/output_opencv_sift_model/images.bin colmap_data/CMU_data/slice3_mnm/live/output_opencv_sift_model/points3D.bin colmap_data/CMU_data/slice3_mnm/avg_descs_xyz_ml.npy

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

# The more you increase the vallues the more images will localise from live and gt
# The randomness is to simulate the COLMAP feature extraction 800 query /2000 recon. used in previous papers
query_features_limit = random.randint(700, 900)
reconstr_features_limit = random.randint(1900, 2200)

def empty_points_3D_txt_file(path):
    open(path, 'w').close()

def arrange_images_txt_file(path):
    with open(path, "r") as f:
        lines = f.readlines()
    with open(path, "w") as f:
        for line in lines:
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

def countDominantOrientations(keypoints):
    domOrientations = np.ones([len(keypoints),1])
    for i in range(len(keypoints)):
        if(domOrientations[i,0] == 1): # if point has not been checked before
            dominantsIndex = np.zeros([len(keypoints),1])
            dominantsIndex[i,0] = 1
            nDominants = 1
        for j in range(i+1, len(keypoints), 1):
            dist = np.abs(keypoints[i].pt[0] - keypoints[j].pt[0]) + np.abs(keypoints[i].pt[1] - keypoints[j].pt[1])
            if(dist == 0.0):
                nDominants +=1
                dominantsIndex[j, 0] = 1
        for k in range(len(dominantsIndex)):
            if(dominantsIndex[k,0] == 1):
                domOrientations[k,0] = nDominants
    return domOrientations

base_path = sys.argv[1] #/home/alex/uni/models_for_match_no_match/CMU_slice_3/
model_base_path = os.path.join(base_path, "base")
model_live_path = os.path.join(base_path, "live")
model_gt_path = os.path.join(base_path, "gt")
images_base_path = os.path.join(model_base_path, "images")
images_live_path = os.path.join(model_live_path, "images")
images_gt_path = os.path.join(model_gt_path, "images")
sift = cv2.SIFT_create()

# look at cmu_sparse_reconstuctor.py, for help

# Note: use images names from database to locate them for opencv feature extraction

# 1 - replace base model features with openCV sift (including matches too)
manually_created_model_txt_path = os.path.join(model_base_path,'empty_model_for_triangulation_txt')  # the "empty model" that will be used to create "opencv_sift_model"
os.makedirs(manually_created_model_txt_path, exist_ok=True)
# set up files as stated online in COLMAP's faq
colmap_model_path = os.path.join(model_base_path, 'model/0')
reconstruction = pycolmap.Reconstruction(colmap_model_path)
# export model to txt
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
    # as in paper ~2000 for map, ~800 for query
    # this might lead to same number of features in the database for the base images.
    # For example if more than reconstr_features_limit are detected for multiple images
    # then they will be reconstr_features_limit for multiple images
    kps = kps[0:reconstr_features_limit]
    des = des[0:reconstr_features_limit]
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

for image_name in image_names:
    # we need to loop though all the live images here (including base too ofc)
    image_id = get_image_id(live_db, image_name)
    # check if image exists in base - which means its keypoints and descs have been already replaced, from previous iterations
    # image_id from live, will return None below, because we want it to, as at this point we
    # did not extract opencv data kps and des for that live image
    base_db_keypoints = get_keypoints_data_and_dominantOrientations(base_db, image_id)
    base_db_descriptors = get_descriptors(base_db, image_id)

    if((base_db_keypoints == None) or (base_db_descriptors == None)): #need to replace the kps and descs then in the live db
        # at this point we are looking at a live image_id only.
        print(f'for image id = {image_id}, kps, des, and dominant_orientations has not be computed so will computer them and replace them in the live db')
        img = cv2.imread(os.path.join(images_live_path, image_name))
        kps_plain = []
        kps, des = sift.detectAndCompute(img,None)
        # as in paper ~2000 for map, ~800 for query
        kps = kps[0:query_features_limit]
        des = des[0:query_features_limit]
        dominant_orientations = countDominantOrientations(kps)

        kps_plain += [[kps[i].pt[0], kps[i].pt[1], kps[i].octave, kps[i].angle, kps[i].size, kps[i].response] for i in range(len(kps))]
        kps_plain = np.array(kps_plain)

        live_db.replace_keypoints(image_id, kps_plain, dominant_orientations)
        live_db.replace_descriptors(image_id, des)
    else:
        # copy from base db the descriptors and keypoints already extracted to the live db
        # no need to re-compute dominant_orientations, sift kps and des
        print(f'for image id = {image_id}, kps, des, and dominant_orientations has already computed so will load them from previous database, i.e. base')
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

for image_name in image_names:
    # we need to loop though all the gt images here (including live and base ofc)
    image_id = get_image_id(gt_db, image_name)
    # check if image exists in live - which means its keypoints and descs have been already replaced, from previous iterations
    # image_id from gt, will return None below, because we want it to, as at this point we
    # did not extract opencv data kps and des for that gt image
    live_db_keypoints = get_keypoints_data_and_dominantOrientations(live_db, image_id)
    live_db_descriptors = get_descriptors(live_db, image_id)

    if((live_db_keypoints == None) or (live_db_descriptors == None)): #need to replace the kps and descs then
        # at this point we are looking at a gt image_id only.
        print(f'for image id = {image_id}, kps, des, and dominant_orientations has not be computed so will computer them and replace them in the gt db')
        img = cv2.imread(os.path.join(images_gt_path, image_name))
        kps_plain = []
        kps, des = sift.detectAndCompute(img,None)
        # as in paper ~2000 for map, ~800 for query
        kps = kps[0:query_features_limit]
        des = des[0:query_features_limit]
        dominant_orientations = countDominantOrientations(kps)

        kps_plain += [[kps[i].pt[0], kps[i].pt[1], kps[i].octave, kps[i].angle, kps[i].size, kps[i].response] for i in range(len(kps))]
        kps_plain = np.array(kps_plain)

        gt_db.replace_keypoints(image_id, kps_plain, dominant_orientations)
        gt_db.replace_descriptors(image_id, des)
    else:
        # copy from live db the descriptors and keypoints already extracted to the gt db
        # no need to re-compute dominant_orientations, sift kps and des
        print(f'for image id = {image_id}, kps, des, and dominant_orientations has already computed so will load them from previous database, i.e. live')
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
colmap.image_registrator(qt_db_path, live_opencv_sift_model_path, os.path.join(model_gt_path, 'output_opencv_sift_model'))

print('Gt Model done!')