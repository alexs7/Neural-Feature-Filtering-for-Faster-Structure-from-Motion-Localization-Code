# This file will load all models from test_match_no_match_tool.py and test_predicting_matchability_tool.py and my NN classifier and
# generate the confusion matrix recall, and precision number, etc csv file
# All the results will be saved under the "ML_data" folder for each dataset.
# Run this alongside the print_model_stats.py file

# The difference from test_match_no_match_tool.py is that I test on my own 3D data which makes more sense. So the test_match_no_match_tool.py is not used anymore
# There is not much difference with test_predicting_matchability_tool.py (training on same data) but I use the python model here only (which performs the same as the C++ tool).
# So let's not use test_predicting_matchability_tool.py anymore.

# Here all the models are test at once so easier to examine results.
# Report:
# 1 - Rotation error
# 2 - Translation error
# 3 - Inliers
# 4 - Outliers
# 5 - mAA
# 6 - Feature Time
# 7 - Images and reprojected points
# 8 - Reprojection Error
# 9 - 3D points mathced to 2D points
# 10 - 3D points not matched to 2D points
# 11 - Sample Consensus Time
# 12 - Reduction Percentage
# 13 - Confusion Matrix
# 14 - Precision
# 15 - Recall
# 16 - F1 Score
# 17 - Balanced Accuracy

# A minor note here is that you can use the GT data definition from the papers to evaluate the models from your paper but, when it comes to benchmarking for your thesis
# you will have to use the 3D data, a SIFT feature <-> matched or not to a 3D point. It does not make sense to compare different ground truth data definitions.
# But! because you use matches from the COLMAP two views geometry table to train both models from the papers it is fine because
# each match in that table is used in the 3D recostruction - so it has most probably a 3D point matched to it.

# Also look at show_results_on_images.py, to see how the models perform on images.
# To run this file you need to have all the models trained and the points averaged descriptors for both the MnM model nd my data COLMAP SIFT.

import csv
import os
from scantools.utils.colmap import read_points3D_binary, read_cameras_binary

from query_image import read_images_binary, get_localised_image_by_names

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import sys
import cv2
import numpy as np
from joblib import load
from sklearn.metrics import classification_report, precision_recall_fscore_support, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, balanced_accuracy_score
from tensorflow import keras
from data import getClassificationDataTestingMnM, getClassificationDataTesting
from parameters import Parameters

# # This method is used to load the models and the test data
# # Its purpose it to avoid duplicate code
# # def extract_test_data_and_models(base_path, images_gt_path, parameters):
# #     print("Loading Models..")
# #     # For NF
# #     nn_model_path = os.path.join(base_path, "ML_data")
# #     classification_model = keras.models.load_model(nn_model_path, compile=False)
# #     # For MnM (2020)
# #     mnm_model_path = os.path.join(base_path, parameters.match_or_no_match_comparison_data, "trained_model_HGE_pairs_no_8000.xml")
# #     mnm_cpp_original_model = cv2.ml.RTrees_load(mnm_model_path)
# #     # For PM (2014)
# #     sklearn_model_path = os.path.join(base_path, parameters.predicting_matchability_comparison_data, f"rforest_3500.joblib")
# #     rf_pm = load(sklearn_model_path)
# #     # binary models results csv path
# #     result_file_output_path = os.path.join(base_path, "ML_data", "binary_models_results.csv")
# #
# #     print("Loading the GT data for PM, MnM, and Nf (or nn), in memory..")
# #     # images_gt_path is needed because of LaMAR keeping its images in a different folder compared to CMU and Retail Shop
# #     test_data_nn = getClassificationDataTesting(base_path, images_gt_path)  # SIFT + XY + RGB + Matched
# #     test_data_mnm = getClassificationDataTestingMnM(os.path.join(base_path, parameters.match_or_no_match_comparison_data),
# #                                                     images_gt_path)  # [x,y,octave,angle,size,response,green_intensity,dominantOrientation,matched]
# #     test_data_pm = np.c_[test_data_nn[:, 0:128], test_data_nn[:, -1]]  # just use test_data_nn, it is faster
# #
# #     return test_data_nn, test_data_mnm, test_data_pm, classification_model, mnm_cpp_original_model, rf_pm, result_file_output_path
# #
# # def benchmark_models(classification_model, mnm_cpp_original_model, rf_pm, test_data_nn, test_data_mnm, test_data_pm, result_file_output_path):
# #
# #     # Start writing in csv file
# #     header = ['Model', 'Precision (Positive Class)', 'Recall (Positive Class)', 'F1 Score', 'TN', 'FN', 'FP', 'TP', 'Balanced Accuracy']
# #     with open(result_file_output_path, 'w', encoding='UTF8') as f:
# #         writer = csv.writer(f)
# #         writer.writerow(header)
# #
# #         X_test = test_data_pm[:, 0:128]
# #         y_true = test_data_pm[:, 128].astype(np.uint8)  # or y_test
# #
# #         writer.writerow(["PM (2014)", " ", " ", " ", " ", " ", " ", " ", " "])
# #         print("Loading PM python models..")
# #         y_pred = pm_model.predict(X_test)
# #
# #         print(classification_report(y_true, y_pred, labels=[0, 1]))
# #         print(precision_recall_fscore_support(y_true, y_pred, labels=[0, 1]))
# #         p_score = precision_score(y_true, y_pred)
# #         r_score = recall_score(y_true, y_pred)
# #         tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
# #         f1_s = f1_score(y_true, y_pred)
# #         bal_acc = balanced_accuracy_score(y_true, y_pred)
# #
# #         tp_perc = tp * 100 / len(X_test)
# #         tn_perc = tn * 100 / len(X_test)
# #         fn_perc = fn * 100 / len(X_test)
# #         fp_perc = fp * 100 / len(X_test)
# #
# #         writer.writerow(["", p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc])
# #         print("--->")
# #
# #         # spacer
# #         writer.writerow([" ", " ", " ", " ", " ", " ", " ", " ", " "])
# #
# #         writer.writerow(["MnM (2020)", " ", " ", " ", " ", " ", " ", " ", " "])
# #         print("Getting MnM test data..")
# #         X_test = test_data_mnm[:, 0:8]
# #         y_true = test_data_mnm[:, 8].astype(np.uint8)  # or y_test
# #
# #         print("Loading MnM C++ models using python..")
# #         _, y_pred = mnm_model.predict(X_test)
# #
# #         print(classification_report(y_true, y_pred, labels=[0, 1]))
# #         print(precision_recall_fscore_support(y_true, y_pred, labels=[0, 1]))
# #         p_score = precision_score(y_true, y_pred)
# #         r_score = recall_score(y_true, y_pred)
# #         tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
# #         f1_s = f1_score(y_true, y_pred)
# #         bal_acc = balanced_accuracy_score(y_true, y_pred)
# #
# #         tp_perc = tp * 100 / len(X_test)
# #         tn_perc = tn * 100 / len(X_test)
# #         fn_perc = fn * 100 / len(X_test)
# #         fp_perc = fp * 100 / len(X_test)
# #
# #         writer.writerow(["", p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc])
# #         print("--->")
# #
# #         # no samples here, just 1 NN
# #         writer.writerow(["NNs", " ", " ", " ", " ", " ", " ", " ", " "])
# #
# #         print("Loading my NN models..")
# #         # PM data here can also be used for my NN as they were trained on the same data
# #         X_test = test_data_pm[:, 0:128]
# #         y_true = test_data_pm[:, 128].astype(np.uint8)  # or y_test
# #
# #         classifier_predictions = classification_nn_model.predict_on_batch(X_test)
# #         matchable_desc_indices = np.where(classifier_predictions > 0.5)[0]
# #         predictions = np.zeros(len(classifier_predictions))
# #         predictions[matchable_desc_indices] = 1
# #         y_pred = predictions.astype(np.uint8)
# #
# #         print(classification_report(y_true, y_pred, labels=[0, 1]))
# #         print(precision_recall_fscore_support(y_true, y_pred, labels=[0, 1]))
# #         p_score = precision_score(y_true, y_pred)
# #         r_score = recall_score(y_true, y_pred)
# #         tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
# #         f1_s = f1_score(y_true, y_pred)
# #         bal_acc = balanced_accuracy_score(y_true, y_pred)
# #
# #         tp_perc = tp * 100 / len(X_test)
# #         tn_perc = tn * 100 / len(X_test)
# #         fn_perc = fn * 100 / len(X_test)
# #         fp_perc = fp * 100 / len(X_test)
# #
# #         writer.writerow(["All", p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc])
# #
# #         print("--->")
# #
# #     print("Done!")
#
# root_path = "/media/iNicosiaData/engd_data/"
# dataset = sys.argv[1] #HGE, CAB, LIN (or Other for CMU, retail shop)
#
# if(dataset == "HGE" or dataset == "CAB" or dataset == "LIN"):
#     base_path = os.path.join(root_path, "lamar", f"{dataset}_colmap_model")
#     print("Base path: " + base_path)
#     parameters = Parameters(base_path)
#     image_files_gt_path = os.path.join(root_path, "lamar/", dataset, "sessions", "query_val_phone", "raw_data") #only gt images
#
#     print("Loading model data..")
#     # load images
#     gt_images_bin = read_images_binary(parameters.gt_model_images_path_mnm)
#     # load points3D
#     gt_points3D_bin = read_points3D_binary(parameters.gt_model_points3D_path_mnm)
#     # load cameras for intrinsics
#     gt_cameras_bin = read_cameras_binary(parameters.gt_model_cameras_path_mnm)
#     # load avg descs
#     np.load(parameters.avg_descs_gt_path_mnm).astype(np.float32)
#     # query images names
#     gt_image_names = np.loadtxt(parameters.query_gt_images_txt_path_mnm, dtype=str)
#
#     localised_qt_images_names = get_localised_image_by_names(gt_image_names, parameters.gt_model_images_path_mnm)  # only gt images (localised only)
#     print(f"Total number of gt localised images: {len(localised_qt_images_names)}")
#     print(f"Total number of gt images in(.txt): {len(gt_image_names)}")
#
#     print("Loading model..")
#     mnm_cpp_original_model = cv2.ml.RTrees_load(parameters.mnm_trained_model_path_mnm)
#
#     # Benchmarking MnM Model
#     print("Benchmarking MnM model..")
#     # [x,y,octave,angle,size,response,green_intensity,dominantOrientation,matched]
#     mnm_path = os.path.join(base_path, parameters.match_or_no_match_comparison_data)
#     # all data, and per image data decs, keypoints etc
#     mnm_test_data, gt_images_data = getClassificationDataTestingMnM(localised_qt_images_names, parameters.gt_db_path_mnm, image_files_gt_path, gt_points3D_bin, gt_images_bin) #remember the points are the MnM triangulated points
#     breakpoint()
#     # At this point you generate the binary classifier stats
#
#     # At this point you get the pose data stats
#
# # if(dataset == "CMU"):
# #     slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
# #                     "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
# #     for slice_name in slices_names:
# #         base_path = os.path.join(root_path, "cmu", f"{slice_name}", "exmaps_data")
# #         print("Base path: " + base_path)
# #         parameters = Parameters(base_path)
# #         image_files_gt_path = os.path.join(base_path, "gt", "images")
# #         test_data_nn, test_data_mnm, test_data_pm, \
# #             classification_model, mnm_cpp_original_model, mnm_cpp_original_model, \
# #             result_file_output_path = extract_test_data_and_models(base_path, image_files_gt_path, parameters)
# #         # write to file results
# #         print("Benchmarking and writing to files..")
# #         # benchmark_models(classification_model, mnm_cpp_original_model, rf_pm, test_data_nn, test_data_mnm, test_data_pm, result_file_output_path)
# #
# # if(dataset == "RetailShop"):
# #     base_path = os.path.join(root_path, "retail_shop", "slice1")
# #     print("Base path: " + base_path)
# #     parameters = Parameters(base_path)
# #     image_files_gt_path = os.path.join(base_path, "gt", "images")
# #     test_data_nn, test_data_mnm, test_data_pm, \
# #         classification_model, mnm_cpp_original_model, mnm_cpp_original_model, \
# #         result_file_output_path = extract_test_data_and_models(base_path, image_files_gt_path, parameters)
# #     # write to file results
# #     print("Benchmarking and writing to files..")
# #     # benchmark_models(classification_model, mnm_cpp_original_model, rf_pm, test_data_nn, test_data_mnm, test_data_pm, result_file_output_path)
