# NOTE: As of 19/11/2022 this file is not used anymore
# # This file was added to create 2D-3D matches for Match or No Match: Keypoint Filtering based on Matching Probability (2020) paper,
# # for Predicting Matchability (2014) paper and for a vanillia RF model. Maybe I can add more models, fuck knows
# import os
# import subprocess
# import sys
# import time
# from itertools import chain
# from os.path import exists
# import cv2
# import numpy as np
# from joblib import load
# from tqdm import tqdm
# from database import COLMAPDatabase
# from query_image import get_image_id, get_localised_image_by_names, load_images_from_text_file, get_keypoints_data, get_keypoints_xy, get_image_name_only
# from save_2D_points import save_debug_image
#
# base_path = sys.argv[1]
# base_path_mnm = sys.argv[2]
#
# images_keypoints_predicted_keypoints_all_models_path = os.path.join(base_path, "images_keypoints_predicted_keypoints_all_models_path")
# if (exists(images_keypoints_predicted_keypoints_all_models_path) == False):
#     print("images_keypoints_predicted_keypoints_all_models_path does not exist! will create")  # same images here will keep be overwritten no need to delete
#     os.makedirs(images_keypoints_predicted_keypoints_all_models_path, exist_ok=True)
#
#
# query_images_path = os.path.join(base_path, "gt/query_name.txt")
# query_images_names = load_images_from_text_file(query_images_path)
# image_gt_dir = os.path.join(base_path_mnm, 'gt/images/')
#
# # OpenCV data for MnM
# query_images_bin_path_mnm = os.path.join(base_path_mnm, "gt/output_opencv_sift_model/images.bin")
# localised_query_images_names_mnm = get_localised_image_by_names(query_images_names, query_images_bin_path_mnm)
#
# # Default data
# query_images_bin_path = os.path.join(base_path, "gt/model/images.bin")
# localised_query_images_names = get_localised_image_by_names(query_images_names, query_images_bin_path)
#
# assert len(localised_query_images_names) != len(localised_query_images_names_mnm)
#
# print("Doing MnM ..")
# model_path_opencv = "/home/Neural-Feature-Filtering-for-Faster-Structure-from-Motion-Localization-Code/code_to_compare/Match-or-no-match-Keypoint-filtering-based-on-matching-probability/build/Trained model.xml"
# model = cv2.ml.RTrees_load(model_path_opencv)
# db_mnm = COLMAPDatabase.connect(os.path.join(base_path_mnm, "gt/database.db"))
#
# for i in tqdm(range(len(localised_query_images_names_mnm))):
#     query_image = localised_query_images_names_mnm[i] #name string
#     image_gt_path = os.path.join(image_gt_dir, query_image)
#     query_image_file = cv2.imread(image_gt_path)  # no need for cv2.COLOR_BGR2RGB here as G is in the middle anw
#     image_id = get_image_id(db_mnm, query_image)
#     keypoints_data = get_keypoints_data(db_mnm, image_id, query_image_file)
#     keypoints_xy = get_keypoints_xy(db_mnm, image_id)
#     _, predictions = model.predict(keypoints_data)  # array of arrays
#     predictions = np.array(predictions.ravel()).astype(np.uint8)
#     save_debug_image(image_gt_path, keypoints_xy, keypoints_xy[predictions == 1],
#                      os.path.join(images_keypoints_predicted_keypoints_all_models_path, get_image_name_only(f"mnm_{query_image}")), query_image)
#
# print("Doing PM ..")
# comparison_data_path_PM = os.path.join(base_path, "predicting_matchability_comparison_data")
# model_path = os.path.join(comparison_data_path_PM, "rforest_all_unbalanced.joblib")
# model = load(model_path)
# db = COLMAPDatabase.connect(os.path.join(base_path, "gt/database.db"))
#
# for i in tqdm(range(len(localised_query_images_names))):
#     query_image = localised_query_images_names[i] #name string
#     image_gt_path = os.path.join(image_gt_dir, query_image)
#     query_image_file = cv2.imread(image_gt_path)  # no need for cv2.COLOR_BGR2RGB here as G is in the middle anw
#     image_id = get_image_id(db, query_image)
#     keypoints_data = get_keypoints_data(db, image_id, query_image_file)
#     keypoints_xy = get_keypoints_xy(db, image_id)
#     _, predictions = model.predict(keypoints_data)  # array of arrays
#     predictions = np.array(predictions.ravel()).astype(np.uint8)
#     save_debug_image(image_gt_path, keypoints_xy, keypoints_xy[predictions == 1],
#                      os.path.join(images_keypoints_predicted_keypoints_all_models_path, get_image_name_only(f"pm_{query_image}")), query_image)
#
# import pdb
# pdb.set_trace()