# NOTE: 03/03/2023 Not using this file anymore
# import os
# import cv2
# from query_image import read_images_binary, get_gt_images_only, get_image_decs, get_keypoints_xy
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
# from database import COLMAPDatabase
# import numpy as np
# import sys
# from joblib import load
#
# # Need to run "prepare_comparison_data.py" before this file, and that the "random_percentage" matches the one from "prepare_comparison_data.py" results
#
# # This script will read anything that has "Extended" in it's name. This was the best performing model name and I kept it. So when you train another model
# # with a different name make sure before you run this script the name is changed to "Extended" - (18/07/2021), TODO: this needs to fixed
# # 08/08/2021 changed to "Extended_New_..."
# # 01/11/2021 changed to "Extended..."
#
# # use CMU_data dir or Coop_data
# # example command (comment and uncomment):
# # "CMU" and "slice3" are the data used for training
# # python3 model_evaluator.py colmap_data/CMU_data/slice3/ CMU slice3 early_stop_model 5
# # "CMU" and "all" are the data used for training
# # python3 model_evaluator.py colmap_data/CMU_data/slice3/ all slice3 early_stop_model 5 -> This is the case for the network trained on all the CMU data
# # or for Coop
# # python3 model_evaluator.py colmap_data/Coop_data/slice1/ Coop slice1 early_stop_model
# # TODO: For this code in this file you have to use the container 'ar2056_bath2020ssh (_ssd)' in weatherwax, ssh root@172.17.0.13 (or whatever IP it is), (updated 19/06/2022)
# # This is because the method predict_on_batch() needs the GPUs for speed - make sure they are free too.
# # If you test multiple datasets, slice4, slice3, run the in sequence as prediction time will be slower if ran in parallel - RUN the evaluator for all datasets on 1 machine!
# # Look at the command below:
# # you can run this on any GPU machine as long as it is free:  ar2056_trainNN_GPU_5 etc etc...
# # python3 model_evaluator.py colmap_data/CMU_data/slice3/ CMU slice3 early_stop_model 5 && python3 model_evaluator.py colmap_data/CMU_data/slice4/ CMU slice4 early_stop_model 5 && python3 model_evaluator.py colmap_data/CMU_data/slice6/ CMU slice6 early_stop_model 5 && python3 model_evaluator.py colmap_data/CMU_data/slice10/ CMU slice10 early_stop_model 5 && python3 model_evaluator.py colmap_data/CMU_data/slice11/ CMU slice11 early_stop_model 5 && python3 model_evaluator.py colmap_data/Coop_data/slice1/ Coop slice1 early_stop_model 5
# # After this script ran: print_eval_NN_results.py to aggregate the results.
#
# # This file will load all gt or query images and their descriptors and necessary data to run the models predictions.
# # For MnM for example it will also load the additional scale / orientation data etc.
# # You need the gt camera pose which you will load from COLMAP gt model.
#
# # Params needed
# # 1. path to model's gt database
# # 2. path to model gt colmap model
# # 3. paths to models to evaluate
#
# base_path = sys.argv[1]
# gt_image_path = sys.argv[2]
#
# # 1. Get data for all methods (except MnM)
#
# # the "gt" here means ground truth (also used as query)
# db_gt_path = os.path.join(base_path, "gt/database.db")
# live_images_bin_path = os.path.join(base_path, "live/model/images.bin")
# query_images_bin_path = os.path.join(base_path, "gt/model/images.bin")
# query_cameras_bin_path = os.path.join(base_path, "gt/model/cameras.bin")
# db_gt = COLMAPDatabase.connect(db_gt_path)  # you need this database to get the query images descs as they do not exist in the live db!
# query_images = read_images_binary(query_images_bin_path)
# live_images = read_images_binary(live_images_bin_path)
# # get gt images only
# gt_only_images = get_gt_images_only(query_images, live_images)
#
# # 2. Get data for MnM
#
# # the "gt" here means ground truth (also used as query)
# base_path_mnm = os.path.join(base_path, "models_for_match_no_match")
# db_gt_path_mnm = os.path.join(base_path_mnm, "gt/database.db")
# live_images_bin_path_mnm = os.path.join(base_path_mnm, "live/model/images.bin")
# query_images_bin_path_mnm = os.path.join(base_path_mnm, "gt/model/images.bin")
# query_cameras_bin_path_mnm = os.path.join(base_path_mnm, "gt/model/cameras.bin")
# db_gt_mnm = COLMAPDatabase.connect(db_gt_path_mnm)  # you need this database to get the query images descs as they do not exist in the live db!
# query_images_mnm = read_images_binary(query_images_bin_path)
# live_images_mnm = read_images_binary(live_images_bin_path)
# # get gt images only
# gt_only_images_mnm = get_gt_images_only(query_images_mnm, live_images_mnm)
#
# print("Loading Predicting Matchability (2014)...")
# pm_comparison_data_path = os.path.join(base_path, "predicting_matchability_comparison_data")
# no_samples = 3500
# pm_model_path = os.path.join(pm_comparison_data_path, f"rforest_{no_samples}.joblib")
# pm_model = load(pm_model_path)
#
# print("Loading Match or No Match: Keypoint Filtering based on Matching Probability + (OpenCV)..")
# model_path = os.path.join(comparison_data_path, f"Trained model {no_samples}.xml")
# model = cv2.ml.RTrees_load(model_path)
#
# # 3. Get loop through all query images and get their descriptors and relevant data
# for img_id , img_data in gt_only_images.item():
#     descs = get_image_decs(db_gt, img_id)
#     keypoints_xy = get_keypoints_xy(db_gt, img_id)
#     assert (img_data.xys.shape[0] == img_data.point3D_ids.shape[0] == descs.shape[0])  # just for my sanity
#     img_file = cv2.imread(os.path.join(os.path.join(gt_image_path, img_data.name)))
#     for i in range(img_data.xys.shape[0]):
#         xy = img_data.xys[i]
#         y = np.round(xy[1]).astype(int)
#         x = np.round(xy[0]).astype(int)
#         if(y >= img_file.shape[0] or x >= img_file.shape[1]):
#             continue
#         brg = img_file[y, x] #opencv conventions
#         blue = brg[0]
#         green = brg[1]
#         red = brg[2]
#         breakpoint()
#
# # Do MnM