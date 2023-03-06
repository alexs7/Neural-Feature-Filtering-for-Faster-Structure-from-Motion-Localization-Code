# This file will load all models from test_match_no_match_tool.py and test_predicting_matchability_tool.py and my NN classifier and
# generate the confusion matrix recall, and precision number, etc csv file
# All the results will be saved under the "ML_data" folder for each dataset.
# Run this alongside the print_model_stats.py file
# The difference from test_match_no_match_tool.py is that I test on my own 3D data which makes more sense. So the test_match_no_match_tool.py is not used anymore
# There is not much difference with test_predicting_matchability_tool.py (training on same data) but I use the python model here only (which performs the same as the C++ tool).
# So let's not use test_predicting_matchability_tool.py anymore.

# 1 - Confusion Matrix
# 2 - Precision
# 3 - Recall
# 4 - F1 Score
# 5 - Balanced Accuracy

# A minor note here is that you can use the GT data definition from the papers to evaluate the models from your paper but, when it comes to benchmarking for your thesis
# you will have to use the 3D data, a SIFT feature <-> matched or not to a 3D point. It does not make sense to compare different ground truth data definitions.
# But! because you use matches from the COLMAP two views geometry table to train both models from the papers it is fine because
# each match in that table is used in the 3D recostruction - so it has most probably a 3D point matched to it.

# Also look at show_results_on_images.py, to see how the models perform on images.
# To run this file you need to have all the models trained and the points averaged descriptors for both the MnM model nd my data COLMAP SIFT.

import csv
import os
import cv2
import numpy as np
from joblib import load
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score, accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import sys
from data import get_MnM_data, get_default_data
from parameters import Parameters
from tensorflow import keras

def get_binary_classifier_results(y_true, y_pred):
    p_score = precision_score(y_true, y_pred)  # tp / (tp + fp)
    p_score = np.round(p_score, 2)
    r_score = recall_score(y_true, y_pred)  # tp / (tp + fn)
    r_score = np.round(r_score, 2)
    f1_s = f1_score(y_true, y_pred)  # 2 * (precision * recall) / (precision + recall)
    f1_s = np.round(f1_s, 2)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tp_perc = np.round(tp * 100 / len(y_true), 2)
    tn_perc = np.round(tn * 100 / len(y_true), 2)
    fn_perc = np.round(fn * 100 / len(y_true), 2)
    fp_perc = np.round(fp * 100 / len(y_true), 2)
    bal_acc = balanced_accuracy_score(y_true, y_pred)  # for imbalanced data
    bal_acc = np.round(bal_acc, 2)
    acc = accuracy_score(y_true, y_pred)
    acc = np.round(acc, 2)
    return f'{p_score:.2f}', f'{r_score:.2f}', f'{f1_s:.2f}', f'{tn_perc:.2f}', f'{fn_perc:.2f}', f'{fp_perc:.2f}', f'{tp_perc:.2f}', f'{bal_acc:.2f}', f'{acc:.2f}'

def write_binary_classifier_results_to_csv(nf_model, pm_model, mnm_model, mnm_gt_data, default_data, writer, dataset):
    # Start writing in csv file
    header = [dataset, 'Precision (Positive Class)', 'Recall (Positive Class)', 'F1 Score', 'TN (%)', 'FN (%)', 'FP (%)', 'TP (%)', 'Balanced Accuracy', 'Accuracy']
    writer.writerow(header)

    nf_data = default_data[:, 0:133] #feature data
    pm_data = default_data[:, 0:128] #feature data
    y_true = default_data[:, 133].astype(np.uint8)  # classes

    # mnm data
    mnm_data = mnm_gt_data[:,0:8].astype(np.float32)
    mnm_y_true = mnm_gt_data[:,8].astype(np.uint8)

    print("Predicting..")
    # predictions
    # This returns a float between 0 and 1
    y_pred_nf = nf_model.predict(nf_data)
    # The ones below return 0 an 1
    y_pred_pm = pm_model.predict(pm_data)
    _, y_pred_mnm = mnm_model.predict(mnm_data)

    print("Writing results..")
    y_pred_nf = np.where(y_pred_nf >= 0.5, 1, 0)
    p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc = get_binary_classifier_results(y_true, y_pred_nf)
    writer.writerow(["NF", p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc])

    p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc = get_binary_classifier_results(y_true, y_pred_pm)
    writer.writerow(["PM (2014)", p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc])

    p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc = get_binary_classifier_results(mnm_y_true, y_pred_mnm)
    writer.writerow(["MnM (2020)", p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc])

    writer.writerow("")

def write_lamar_res(writer):
    root_path = "/media/iNicosiaData/engd_data/"
    for dataset in ["HGE", "CAB", "LIN"]:
        base_path = os.path.join(root_path, "lamar", f"{dataset}_colmap_model")
        print("Base path: " + base_path)
        parameters = Parameters(base_path)

        # load data MnM
        print("Getting MnM data..")
        mnm_gt_data = get_MnM_data(parameters)

        print("Getting NN and PM data")
        # sift + xy + rgb
        gt_image_path = os.path.join(root_path, "lamar", dataset, "sessions", "query_val_phone", "raw_data")
        default_data = get_default_data(parameters, gt_image_path)

        # load models
        # For NF
        nn_model_path = os.path.join(base_path, "ML_data", "classification_model")
        nf_model = keras.models.load_model(nn_model_path, compile=False)
        print(nf_model.summary())
        # For MnM (2020)
        mnm_model_path = os.path.join(base_path, parameters.mnm_path, "trained_model_pairs_no_8000.xml")
        mnm_model = cv2.ml.RTrees_load(mnm_model_path)
        # For PM (2014)
        pm_model_path = os.path.join(base_path, parameters.predicting_matchability_comparison_data, f"rforest_3500.joblib")
        pm_model = load(pm_model_path)

        write_binary_classifier_results_to_csv(nf_model, pm_model, mnm_model, mnm_gt_data, default_data, writer, dataset)

def write_cmu_res(writer):
    root_path = "/media/iNicosiaData/engd_data/"
    slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                    "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    for slice_name in slices_names:
        base_path = os.path.join(root_path, "cmu", f"{slice_name}", "exmaps_data")
        print("Base path: " + base_path)
        parameters = Parameters(base_path)

        # load data MnM
        print("Getting MnM data..")
        mnm_gt_data = get_MnM_data(parameters)

        print("Getting NN and PM data")
        # sift + xy + rgb
        gt_image_path = os.path.join(base_path, "gt", "images")
        default_data = get_default_data(parameters, gt_image_path)

        # load models
        # For NF
        nn_model_path = os.path.join(base_path, "ML_data", "classification_model")
        nf_model = keras.models.load_model(nn_model_path, compile=False)
        print(nf_model.summary())
        # For MnM (2020)
        mnm_model_path = os.path.join(base_path, parameters.mnm_path, "trained_model_pairs_no_4000.xml")
        mnm_model = cv2.ml.RTrees_load(mnm_model_path)
        # For PM (2014)
        pm_model_path = os.path.join(base_path, parameters.predicting_matchability_comparison_data, f"rforest_1500.joblib")
        pm_model = load(pm_model_path)

        write_binary_classifier_results_to_csv(nf_model, pm_model, mnm_model, mnm_gt_data, default_data, writer, slice_name)

def write_retail_shop_res(writer):
    root_path = "/media/iNicosiaData/engd_data/"
    base_path = os.path.join(root_path, "retail_shop", "slice1")
    dataset = "RetailShop"
    print("Base path: " + base_path)
    parameters = Parameters(base_path)

    # load data MnM
    print("Getting MnM data..")
    mnm_gt_data = get_MnM_data(parameters)

    print("Getting NN and PM data")
    # sift + xy + rgb
    gt_image_path = os.path.join(base_path, "gt", "images")
    default_data = get_default_data(parameters, gt_image_path)

    # load models
    # For NF
    nn_model_path = os.path.join(base_path, "ML_data", "classification_model")
    nf_model = keras.models.load_model(nn_model_path, compile=False)
    print(nf_model.summary())
    # For MnM (2020)
    mnm_model_path = os.path.join(base_path, parameters.mnm_path, "trained_model_pairs_no_4000.xml")
    mnm_model = cv2.ml.RTrees_load(mnm_model_path)
    # For PM (2014)
    pm_model_path = os.path.join(base_path, parameters.predicting_matchability_comparison_data, f"rforest_3000.joblib")
    pm_model = load(pm_model_path)

    write_binary_classifier_results_to_csv(nf_model, pm_model, mnm_model, mnm_gt_data, default_data, writer, dataset)

root_path = "/media/iNicosiaData/engd_data/"
result_file_output_path = os.path.join(root_path, "binary_classifier_results.csv")

with open(result_file_output_path, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    # write_cmu_res(writer)
    # write_retail_shop_res(writer)
    write_lamar_res(writer)
