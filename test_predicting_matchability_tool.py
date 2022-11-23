# This file will test the original random forest tool from PM paper, and the sklearn
# given a model name, for example:  rforest_1500_balanced, rforest_500_weighted, rforest_all_unbalanced
# It will load a model from code_to_compare/Predicting_Matchability/rforest, and
# colmap_data/CMU_data/slice3/predicting_matchability_comparison_data/ and run the model on the TEST data in "code_to_compare/Predicting_Matchability/rforest"

# NOTE: 23/11/2022: Use test_all_models_on_3D_gt_data.py instead

# import os
# import subprocess
# import sys
# import numpy as np
# from generate_gt_truth_testing import createGTTestDataForPM
# from sklearn.metrics import classification_report, precision_recall_fscore_support
# from sklearn.metrics import confusion_matrix
# from joblib import load
#
# def write_test_to_file(test_data, original_too_test_data_path):
#     print(f"Test data size: {test_data.shape[0]}")
#     with open(original_too_test_data_path, 'w') as f:
#         for desc in test_data:
#             row = ' '.join([str(num) for num in desc[0:128].astype(np.uint8)])
#             f.write(f"{row}\n")
#     print("Test data saved!")
#
# def load_predictions(original_output_results_path):
#     res = np.loadtxt(original_output_results_path)
#     y_pred = np.empty([res.shape[0]])
#     for i in range(res.shape[0]):
#         el = res[i]
#         pos = el[1]
#         if (pos >= 0.5):
#             y_pred[i] = 1
#         else:
#             y_pred[i] = 0
#     return y_pred
#
# print("Testing Original Tool. Reminder, did you train the tool first ?")
# base_path = sys.argv[1]
# print("Base path: " + base_path)
# no_samples = sys.argv[2]
# sklearn_model_path = os.path.join(os.path.join(base_path, "predicting_matchability_comparison_data"), f"rforest_{no_samples}.joblib")
# original_tool_path = "/home/Neural-Feature-Filtering-for-Faster-Structure-from-Motion-Localization-Code/code_to_compare/Predicting_Matchability/rforest"
# original_too_test_data_path = os.path.join(f"{original_tool_path}", f"descs_test.txt")
#
# print("Getting test data and writing to file..")
# test_data = createGTTestDataForPM(base_path) # N x 129
# write_test_to_file(test_data, original_too_test_data_path) #for the original tool, only saves the SIFT vecs
#
# original_rforest_model_path = os.path.join(original_tool_path, f"rforest_{no_samples}.gz")
# original_output_results_path = os.path.join(original_tool_path, f"{no_samples}_results.txt") #results here
# original_tool_predict_command = os.path.join(original_tool_path, "./rforest")
#
# print("Running original tool..")
# # rforest.exe -f rforest.gz -i desc.txt -o res.txt
# # this will also produce a test_time.txt file in the home dir (Neural Filter .. dir)
# original_tool_exec = [original_tool_predict_command, "-f", original_rforest_model_path, "-i", original_too_test_data_path, "-o", original_output_results_path]
# subprocess.check_call(original_tool_exec)
#
# print("Done!")
#
# # (the first and the second columns correspond to labels 0 and 1, respectively)
# y_pred = load_predictions(original_output_results_path)
# y_true = test_data[:,128]
#
# print("Classification_report")
# print(classification_report(y_true, y_pred, labels=[0, 1]))
# print("Precision_recall_fscore_support")
# print(precision_recall_fscore_support(y_true, y_pred, labels=[0, 1]))
# print("Confusion_matrix")
# print(confusion_matrix(y_true, y_pred))
#
# print("Testing sklearn RF... Reminder, did you train the tool first ?")
# rf = load(sklearn_model_path)
# y_pred = rf.predict(test_data[:,0:128])
#
# print("Classification_report")
# print(classification_report(y_true, y_pred, labels=[0, 1]))
# print("Precision_recall_fscore_support")
# print(precision_recall_fscore_support(y_true, y_pred, labels=[0, 1]))
# print("Confusion_matrix")
# print(confusion_matrix(y_true, y_pred))
#
# print("Done!")