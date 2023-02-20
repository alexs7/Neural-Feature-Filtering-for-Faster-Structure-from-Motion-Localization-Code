# Run this after format_data_for_match_no_match.py and get_points_3D_mean_desc_single_model_ml_mnm.py
# and before create_training_data_and_train_for_match_no_match.py.
# This file will create and define the GT to use later for testing once you have the trained model ready.

# NOTE - 20/02/2023 - This file is not needed anymore. You can use your own GT 3D data for testing.
# To explain you can use this file if you want to benchmark the MnM model (c++) on the same data structure as it was trained.
# But the test data for this piece of work is to run the models on my 3D test data.

# import os
# import shutil
# import subprocess
# import sys
# import numpy as np
# from tqdm import tqdm
# from database import COLMAPDatabase
# from database import pair_id_to_image_ids
# from query_image import load_images_from_text_file, get_image_name_from_db_with_id
#
# def clear_folder(folder_path):
#     print(f"Deleting {folder_path}")
#     if (os.path.exists(folder_path)):
#         shutil.rmtree(folder_path)
#     os.makedirs(folder_path, exist_ok=True)
#     pass
#
# def createGroundTruthDataForMatchNoMatchMatchabilityComparison(mnm_base_path, mnm_base_code_dir, gt_images_path, data_path):
#     # clear folders
#     images_for_original_code = os.path.join(mnm_base_code_dir, "Training images")  # same name as in MnM code base main.cpp
#     clear_folder(images_for_original_code)
#     # from the original code "Training Data" will contain the data after the pre-training stage
#     # Each time you run the original pre-training (C++) you have to empty the "Training Data" folder, done here
#     training_data_csv_for_original_code = os.path.join(mnm_base_code_dir, "Training Data")
#     clear_folder(training_data_csv_for_original_code)
#     # This will contain Ground Truth data generated from the C++ pretraining tool. SAme number of images as in "Training images"
#     ground_truth_data_from_original_code = os.path.join(mnm_base_code_dir, "Ground Truth")
#     clear_folder(ground_truth_data_from_original_code)
#     # clear test images
#     test_images_for_original_code = os.path.join(mnm_base_code_dir, "Test Images")
#     clear_folder(test_images_for_original_code)
#
#     db_gt_mnm_path = os.path.join(mnm_base_path, "gt/database.db")  # openCV db + extra MnM data
#     db_gt_mnm = COLMAPDatabase.connect(db_gt_mnm_path)  # remember this database holds the OpenCV descriptors
#     query_images_path = os.path.join(base_path, "gt/query_name.txt")  # these are the same anw, as mnm
#     query_images_names = load_images_from_text_file(query_images_path)
#
#     # get all pair ids from gt database
#     all_pairs_ids = db_gt_mnm.execute("SELECT pair_id FROM two_view_geometries").fetchall()
#     all_pairs_ids = np.array(all_pairs_ids)
#     image_id_l, image_id_r = pair_id_to_image_ids(all_pairs_ids)
#     all_img_ids_from_pairs = np.c_[image_id_l, image_id_r]
#
#     # get all image ids only of query_images_names
#     gt_image_ids = []
#     for name in query_images_names:
#         id = db_gt_mnm.execute("SELECT image_id FROM images WHERE name = " + "'" + str(name) + "'").fetchone()[0]
#         gt_image_ids.append(id)
#
#     # Here I set my own ground truth images. On the first occurrence of a pair I save the images in the "Training Data" folder
#     # Note that here I pick the first occurring pair, for example [2499.0, 2508.0]. If [2499.0, 2291.0] then shows up,
#     # then I won't save the latter two images. This is because if I save both then I will have ground truth ambiguity when testing,
#     # i.e. will have two images for 2499.0 with different gt data. So the problem will be which one will I choose ?
#     file_index = 1
#     gt_images = []
#     images_saved = [] #just to keep track
#     for ids in tqdm(all_img_ids_from_pairs): # here we only want pairs that contain two ids from the gt images
#         if(ids[0] in gt_image_ids and ids[1] in gt_image_ids):
#             name_l = get_image_name_from_db_with_id(db_gt_mnm, ids[0])
#             name_r = get_image_name_from_db_with_id(db_gt_mnm, ids[1])
#             # save the first occurring pair
#             # only save if both images do not already exist
#             if((name_l not in images_saved) and (name_r not in images_saved)):
#                 shutil.copyfile(os.path.join(gt_images_path, name_l), os.path.join(images_for_original_code, f"image_{'%010d' % file_index}.jpg"))
#                 shutil.copyfile(os.path.join(gt_images_path, name_r), os.path.join(images_for_original_code, f"image_{'%010d' % (file_index + 1)}.jpg"))
#                 # so to avoid duplicates
#                 gt_images.append([name_l, name_r])
#                 images_saved.append(name_l)
#                 images_saved.append(name_r)
#                 file_index += 2
#
#     # At this point now I have the gt pairs in "Training Data" and I will run the C++ tool to generate the GT data
#     matchornomatch_gt = ["./matchornomatch_gt"]
#     subprocess.check_call(matchornomatch_gt , cwd=mnm_base_code_dir)
#
#     # Now you will need to copy the Gt data to the MnM comparison folder
#     mnm_gt_comparison_folder_path = os.path.join(data_path, "ground_truth_from_cpp_tool")
#     clear_folder(mnm_gt_comparison_folder_path)
#     # C++ tool uses this order:
#     # image_0000000001.jpg and Training images/image_0000000002.jpg,
#     # image_0000000003.jpg and Training images/image_0000000004.jpg ...
#     # By looking at how images are saved and added in gt_images, each pair is name_l, name_r ...
#     # so for i=0: Training images/image_0000000001.jpg and Training images/image_0000000002.jpg is gt_images[0][0], gt_images[0][1], is Unbalanced S 1, Unbalanced T 1
#     for i in tqdm(range(len(gt_images))):
#         s_name = f'{gt_images[i][0].split("/")[0].split(".")[0]}.csv'
#         shutil.copyfile(os.path.join(ground_truth_data_from_original_code, f"Unbalanced S {i+1}"), os.path.join(mnm_gt_comparison_folder_path, s_name))
#         t_name = f'{gt_images[i][1].split("/")[0].split(".")[0]}.csv'
#         shutil.copyfile(os.path.join(ground_truth_data_from_original_code, f"Unbalanced T {i+1}"), os.path.join(mnm_gt_comparison_folder_path, t_name))
#     print("Done!")
#
# base_path = sys.argv[1]
# mnm_base_path = sys.argv[2] # this is that data generated from format_data_for_match_no_match.py
# gt_images_path = sys.argv[3] #i.e. colmap_data/CMU_data/slice3/gt/images/session_7
# data_path = os.path.join(base_path, "match_or_no_match_comparison_data")
# # This will contain images and csv files to use for the MnM code!
# mnm_base_code_dir = "code_to_compare/Match-or-no-match-Keypoint-filtering-based-on-matching-probability/build/"
# os.makedirs(data_path, exist_ok = True)
#
# createGroundTruthDataForMatchNoMatchMatchabilityComparison(mnm_base_path, mnm_base_code_dir, gt_images_path, data_path)


