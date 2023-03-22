# This file reports binary stats:

# 1 - Confusion Matrix
# 2 - Precision
# 3 - Recall
# 4 - F1 Score
# 5 - Balanced Accuracy

# The result .csv file can be parsed with parse_results_for_thesis.py
# NOTE: I trained a model on SIFT+RGB values only nf_small, but I am not using it now.

import csv
import os
import sys

import cv2
import numpy as np
from joblib import load
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score, accuracy_score
from database import COLMAPDatabase
from query_image import get_test_data_all
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
from parameters import Parameters
from tensorflow import keras

# mnm_model_name, pm_model_name are passed as they differ per dataset
# nf is the same for all datasets
def load_models(parameters, mnm_model_name, pm_model_name):
    # load models
    print("Loading models..")
    # For MnM (2020)
    mnm_model_path = os.path.join(parameters.base_path, parameters.mnm_path, mnm_model_name)
    mnm_model = cv2.ml.RTrees_load(mnm_model_path)
    # For NF (2023)
    nn_model_path = os.path.join(parameters.base_path, "ML_data", "classification_model")
    nf_model = keras.models.load_model(nn_model_path, compile=False)
    # This model only is trained on SIFT XY BRG data
    nn_model_small_path = os.path.join(parameters.base_path, "ML_data", "classification_model_small")
    nf_model_small = keras.models.load_model(nn_model_small_path, compile=False)
    # NF BCE variation
    nn_model_bce_path = os.path.join(parameters.base_path, "ML_data", "classification_model_bce")
    nf_model_bce = keras.models.load_model(nn_model_bce_path, compile=False)
    # For PM (2014)
    pm_model_path = os.path.join(parameters.base_path, parameters.predicting_matchability_comparison_data, pm_model_name)
    pm_model = load(pm_model_path)

    models = {}
    models["mnm_model"] = mnm_model
    models["nf_model"] = nf_model
    models["nf_model_small"] = nf_model_small
    models["nf_model_bce"] = nf_model_bce
    models["pm_model"] = pm_model

    return models

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
    # str: f'{p_score:.2f}', f'{r_score:.2f}', f'{f1_s:.2f}', f'{tn_perc:.2f}', f'{fn_perc:.2f}', f'{fp_perc:.2f}', f'{tp_perc:.2f}', f'{bal_acc:.2f}', f'{acc:.2f}'
    return p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc

def write_binary_classifier_results_to_csv(models, test_data, writer, dataset):

    all_vals = np.empty((5, 9))

    # Start writing in csv file
    header = [dataset, 'Precision (Positive Class)', 'Recall (Positive Class)', 'F1 Score', 'TN (%)', 'FN (%)', 'FP (%)', 'TP (%)', 'Balanced Accuracy', 'Accuracy']
    writer.writerow(header)

    mnm_data = np.c_[test_data[:, 128:130], test_data[:, 133], test_data[:, 134], test_data[:, 135], test_data[:, 136], test_data[:, 131], test_data[:, 137], test_data[:, 138]]
    pm_data = np.c_[test_data[:, 128:130], test_data[:, 0:128]]
    nf_data = test_data #just for naming purposes
    y_true = test_data[:, -1].astype(np.uint8)  # classes

    mnm_model = models['mnm_model']
    nf_model = models['nf_model']
    nf_model_small = models['nf_model_small']
    nf_model_bce = models['nf_model_bce']
    pm_model = models['pm_model']

    print("Predicting and writing results..")
    # Mnm
    mnm_data = mnm_data[:, 0:8].astype(np.float32)
    _, y_pred_mnm = mnm_model.predict(mnm_data)
    p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc = get_binary_classifier_results(y_true, y_pred_mnm)
    writer.writerow(["MnM", p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc])
    print(f"MnM: {p_score:.3f}, {r_score:.3f}, {f1_s:.3f}, {tn_perc:.3f}, {fn_perc:.3f}, {fp_perc:.3f}, {tp_perc:.3f}, {bal_acc:.3f}, {acc:.3f}")
    all_vals[0,0] = p_score
    all_vals[0,1] = r_score
    all_vals[0,2] = f1_s
    all_vals[0,3] = tn_perc
    all_vals[0,4] = fn_perc
    all_vals[0,5] = fp_perc
    all_vals[0,6] = tp_perc
    all_vals[0,7] = bal_acc
    all_vals[0,8] = acc

    # NF
    prediction_data = nf_data[:, 0:138]  # exclude matched
    y_pred_nf = nf_model.predict(prediction_data, verbose=0)  # returns a value from (0,1)
    y_pred_nf = np.where(y_pred_nf >= 0.5, 1, 0)
    p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc = get_binary_classifier_results(y_true, y_pred_nf)
    writer.writerow(["NF", p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc])
    print(f"NF: {p_score:.3f}, {r_score:.3f}, {f1_s:.3f}, {tn_perc:.3f}, {fn_perc:.3f}, {fp_perc:.3f}, {tp_perc:.3f}, {bal_acc:.3f}, {acc:.3f}")
    all_vals[1,0] = p_score
    all_vals[1,1] = r_score
    all_vals[1,2] = f1_s
    all_vals[1,3] = tn_perc
    all_vals[1,4] = fn_perc
    all_vals[1,5] = fp_perc
    all_vals[1,6] = tp_perc
    all_vals[1,7] = bal_acc
    all_vals[1,8] = acc

    # 07/04/2023 - Added new MSE and BCE variations of NF (prediction data can be the same as above)
    # # NF (MSE)
    # y_pred_nf = nf_model_mse.predict(prediction_data, verbose=0)  # returns a value from (0,1)
    # y_pred_nf = np.where(y_pred_nf >= 0.5, 1, 0)
    # p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc = get_binary_classifier_results(y_true, y_pred_nf, )
    # writer.writerow(["NF (MSE)", p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc])
    # print(f"NF (MSE): {p_score}, {r_score}, {f1_s}, {tn_perc}, {fn_perc}, {fp_perc}, {tp_perc}, {bal_acc}, {acc}")
    # all_vals[2,0] = p_score
    # all_vals[2,1] = r_score
    # all_vals[2,2] = f1_s
    # all_vals[2,3] = tn_perc
    # all_vals[2,4] = fn_perc
    # all_vals[2,5] = fp_perc
    # all_vals[2,6] = tp_perc
    # all_vals[2,7] = bal_acc
    # all_vals[2,8] = acc

    # NF (BCE)
    y_pred_nf = nf_model_bce.predict(prediction_data, verbose=0)  # returns a value from (0,1)
    y_pred_nf = np.where(y_pred_nf >= 0.5, 1, 0)
    p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc = get_binary_classifier_results(y_true, y_pred_nf)
    writer.writerow(["NF (BCE)", p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc])
    print(f"NF (BCE): {p_score:.3f}, {r_score:.3f}, {f1_s:.3f}, {tn_perc:.3f}, {fn_perc:.3f}, {fp_perc:.3f}, {tp_perc:.3f}, {bal_acc:.3f}, {acc:.3f}")
    all_vals[2,0] = p_score
    all_vals[2,1] = r_score
    all_vals[2,2] = f1_s
    all_vals[2,3] = tn_perc
    all_vals[2,4] = fn_perc
    all_vals[2,5] = fp_perc
    all_vals[2,6] = tp_perc
    all_vals[2,7] = bal_acc
    all_vals[2,8] = acc

    # NF (small)
    prediction_data = nf_data[:, 0:133]  # SIFT + XY + BRG
    y_pred_nf_small = nf_model_small.predict(prediction_data, verbose=0)  # returns a value from (0,1)
    y_pred_nf_small = np.where(y_pred_nf_small >= 0.5, 1, 0)
    p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc = get_binary_classifier_results(y_true, y_pred_nf_small)
    writer.writerow(["NF (small)", p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc])
    print(f"NF (small): {p_score:.3f}, {r_score:.3f}, {f1_s:.3f}, {tn_perc:.3f}, {fn_perc:.3f}, {fp_perc:.3f}, {tp_perc:.3f}, {bal_acc:.3f}, {acc:.3f}")
    all_vals[3,0] = p_score
    all_vals[3,1] = r_score
    all_vals[3,2] = f1_s
    all_vals[3,3] = tn_perc
    all_vals[3,4] = fn_perc
    all_vals[3,5] = fp_perc
    all_vals[3,6] = tp_perc
    all_vals[3,7] = bal_acc
    all_vals[3,8] = acc

    # PM
    y_pred_pm = pm_model.predict(pm_data[:, 2:130])
    p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc = get_binary_classifier_results(y_true, y_pred_pm)
    writer.writerow(["PM", p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc])
    print(f"PM: {p_score:.3f}, {r_score:.3f}, {f1_s:.3f}, {tn_perc:.3f}, {fn_perc:.3f}, {fp_perc:.3f}, {tp_perc:.3f}, {bal_acc:.3f}, {acc:.3f}")
    all_vals[4,0] = p_score
    all_vals[4,1] = r_score
    all_vals[4,2] = f1_s
    all_vals[4,3] = tn_perc
    all_vals[4,4] = fn_perc
    all_vals[4,5] = fp_perc
    all_vals[4,6] = tp_perc
    all_vals[4,7] = bal_acc
    all_vals[4,8] = acc

    writer.writerow("")

    return all_vals

def write_lamar_res(writer, writer_avg, root_path):

    all_datasets_vals = []
    for dataset in ["HGE", "CAB", "LIN"]:
        base_path = os.path.join(root_path, "lamar", f"{dataset}_colmap_model")
        print("Base path: " + base_path)
        parameters = Parameters(base_path)
        # get data
        opencv_data_db = COLMAPDatabase.connect(parameters.ml_database_all_opencv_sift_path)
        test_data = get_test_data_all(opencv_data_db)
        pm_model_name = f"rforest_{parameters.predicting_matchability_comparison_data_lamar_no_samples}.joblib"
        mnm_model_name = "trained_model_pairs_no_10000.xml"
        models = load_models(parameters, mnm_model_name, pm_model_name)
        # all vals contain the binary classifier metrics for each sub-dataset, CAB, LIN etc
        vals = write_binary_classifier_results_to_csv(models, test_data, writer, dataset)
        all_datasets_vals.append(vals)
        writer.writerow("")

    # write average of all slices
    # NOTE writing to the writer_avg not writer
    writer_avg.writerow(["Averages LaMAR", 'Precision (Positive Class)', 'Recall (Positive Class)', 'F1 Score', 'TN (%)', 'FN (%)', 'FP (%)', 'TP (%)', 'Balanced Accuracy', 'Accuracy'])
    mnm_vals = np.empty([0,9])
    nf_vals = np.empty([0,9])
    nf_bce_vals = np.empty([0,9])
    nf_small_vals = np.empty([0,9])
    pm_vals = np.empty([0,9])
    for i in range(len(all_datasets_vals)): #same order as in the slices
        subdataset_vals = all_datasets_vals[i] # 5x9, one row for each method, with vals p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc
        mnm_vals = np.r_[mnm_vals, subdataset_vals[0, :].reshape(1,9)]
        nf_vals = np.r_[nf_vals, subdataset_vals[1, :].reshape(1,9)]
        nf_bce_vals = np.r_[nf_bce_vals, subdataset_vals[2, :].reshape(1,9)]
        nf_small_vals = np.r_[nf_small_vals, subdataset_vals[3, :].reshape(1,9)]
        pm_vals = np.r_[pm_vals, subdataset_vals[4, :].reshape(1,9)]

    mnm_vals = np.mean(mnm_vals, axis=0)
    writer_avg.writerow(["MnM", mnm_vals[0], mnm_vals[1], mnm_vals[2], mnm_vals[3], mnm_vals[4], mnm_vals[5], mnm_vals[6], mnm_vals[7], mnm_vals[8]])
    nf_vals = np.mean(nf_vals, axis=0)
    writer_avg.writerow(["NF", nf_vals[0], nf_vals[1], nf_vals[2], nf_vals[3], nf_vals[4], nf_vals[5], nf_vals[6], nf_vals[7], nf_vals[8]])
    nf_bce_vals = np.mean(nf_bce_vals, axis=0)
    writer_avg.writerow(["NF (BCE)", nf_bce_vals[0], nf_bce_vals[1], nf_bce_vals[2], nf_bce_vals[3], nf_bce_vals[4], nf_bce_vals[5], nf_bce_vals[6], nf_bce_vals[7], nf_bce_vals[8]])
    nf_small_vals = np.mean(nf_small_vals, axis=0)
    writer_avg.writerow(["NF (small)", nf_small_vals[0], nf_small_vals[1], nf_small_vals[2], nf_small_vals[3], nf_small_vals[4], nf_small_vals[5], nf_small_vals[6], nf_small_vals[7], nf_small_vals[8]])
    pm_vals = np.mean(pm_vals, axis=0)
    writer_avg.writerow(["PM", pm_vals[0], pm_vals[1], pm_vals[2], pm_vals[3], pm_vals[4], pm_vals[5], pm_vals[6], pm_vals[7], pm_vals[8]])
    writer_avg.writerow("")

def write_cmu_res(writer, writer_avg, root_path):

    slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                    "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    all_slices_vals = []
    for slice_name in slices_names:
        base_path = os.path.join(root_path, "cmu", f"{slice_name}", "exmaps_data")
        print("Base path: " + base_path)
        parameters = Parameters(base_path)
        # get data
        opencv_data_db = COLMAPDatabase.connect(parameters.ml_database_all_opencv_sift_path)
        test_data = get_test_data_all(opencv_data_db)
        pm_model_name = "rforest_1200.joblib"
        mnm_model_name = "trained_model_pairs_no_4000.xml"
        models = load_models(parameters, mnm_model_name, pm_model_name)
        # all vals contain the binary classifier metrics for each slice
        slice_vals = write_binary_classifier_results_to_csv(models, test_data, writer, slice_name)
        all_slices_vals.append(slice_vals)
        writer.writerow("")

    # write average of all slices
    # NOTE writing to the writer_avg not writer
    writer_avg.writerow(["Averages CMU", 'Precision (Positive Class)', 'Recall (Positive Class)', 'F1 Score', 'TN (%)', 'FN (%)', 'FP (%)', 'TP (%)', 'Balanced Accuracy', 'Accuracy'])
    mnm_vals = np.empty([0,9])
    nf_vals = np.empty([0,9])
    nf_bce_vals = np.empty([0,9])
    nf_small_vals = np.empty([0,9])
    pm_vals = np.empty([0,9])
    for i in range(len(all_slices_vals)): #same order as in the slices
        slice_vals = all_slices_vals[i] # 5x9, one row for each method, with vals p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc, acc
        mnm_vals = np.r_[mnm_vals, slice_vals[0, :].reshape(1,9)]
        nf_vals = np.r_[nf_vals, slice_vals[1, :].reshape(1,9)]
        nf_bce_vals = np.r_[nf_bce_vals, slice_vals[2, :].reshape(1,9)]
        nf_small_vals = np.r_[nf_small_vals, slice_vals[3, :].reshape(1,9)]
        pm_vals = np.r_[pm_vals, slice_vals[4, :].reshape(1,9)]

    mnm_vals = np.mean(mnm_vals, axis=0)
    writer_avg.writerow(["MnM", mnm_vals[0], mnm_vals[1], mnm_vals[2], mnm_vals[3], mnm_vals[4], mnm_vals[5], mnm_vals[6], mnm_vals[7], mnm_vals[8]])
    nf_vals = np.mean(nf_vals, axis=0)
    writer_avg.writerow(["NF", nf_vals[0], nf_vals[1], nf_vals[2], nf_vals[3], nf_vals[4], nf_vals[5], nf_vals[6], nf_vals[7], nf_vals[8]])
    nf_bce_vals = np.mean(nf_bce_vals, axis=0)
    writer_avg.writerow(["NF (BCE)", nf_bce_vals[0], nf_bce_vals[1], nf_bce_vals[2], nf_bce_vals[3], nf_bce_vals[4], nf_bce_vals[5], nf_bce_vals[6], nf_bce_vals[7], nf_bce_vals[8]])
    nf_small_vals = np.mean(nf_small_vals, axis=0)
    writer_avg.writerow(["NF (small)", nf_small_vals[0], nf_small_vals[1], nf_small_vals[2], nf_small_vals[3], nf_small_vals[4], nf_small_vals[5], nf_small_vals[6], nf_small_vals[7],nf_small_vals[8]])
    pm_vals = np.mean(pm_vals, axis=0)
    writer_avg.writerow(["PM", pm_vals[0], pm_vals[1], pm_vals[2], pm_vals[3], pm_vals[4], pm_vals[5], pm_vals[6], pm_vals[7], pm_vals[8]])
    writer_avg.writerow("")

def write_retail_shop_res(writer, writer_avg, root_path):

    base_path = os.path.join(root_path, "retail_shop", "slice1")
    dataset = "RetailShop"
    print("Base path: " + base_path)
    parameters = Parameters(base_path)
    # get data
    opencv_data_db = COLMAPDatabase.connect(parameters.ml_database_all_opencv_sift_path)
    test_data = get_test_data_all(opencv_data_db)
    pm_model_name = "rforest_1200.joblib"
    mnm_model_name = "trained_model_pairs_no_4000.xml"
    models = load_models(parameters, mnm_model_name, pm_model_name)
    # all vals contain the binary classifier metrics for each dataset, just retail here
    vals = write_binary_classifier_results_to_csv(models, test_data, writer, dataset)
    writer.writerow("")

    # NOTE writing to the writer_avg not writer
    writer_avg.writerow(["Averages Retail Shop", 'Precision (Positive Class)', 'Recall (Positive Class)', 'F1 Score', 'TN (%)', 'FN (%)', 'FP (%)', 'TP (%)', 'Balanced Accuracy', 'Accuracy']) #no need to average, just one dataset
    mnm_vals = vals[0, :]
    writer_avg.writerow(["MnM", mnm_vals[0], mnm_vals[1], mnm_vals[2], mnm_vals[3], mnm_vals[4], mnm_vals[5], mnm_vals[6], mnm_vals[7], mnm_vals[8]])
    nf_vals = vals[1, :]
    writer_avg.writerow(["NF", nf_vals[0], nf_vals[1], nf_vals[2], nf_vals[3], nf_vals[4], nf_vals[5], nf_vals[6], nf_vals[7], nf_vals[8]])
    nf_bce_vals = vals[2, :]
    writer_avg.writerow(["NF (BCE)", nf_bce_vals[0], nf_bce_vals[1], nf_bce_vals[2], nf_bce_vals[3], nf_bce_vals[4], nf_bce_vals[5], nf_bce_vals[6], nf_bce_vals[7], nf_bce_vals[8]])
    nf_small_vals = vals[3, :]
    writer_avg.writerow(["NF (small)", nf_small_vals[0], nf_small_vals[1], nf_small_vals[2], nf_small_vals[3], nf_small_vals[4], nf_small_vals[5], nf_small_vals[6], nf_small_vals[7],nf_small_vals[8]])
    pm_vals = vals[4, :]
    writer_avg.writerow(["PM", pm_vals[0], pm_vals[1], pm_vals[2], pm_vals[3], pm_vals[4], pm_vals[5], pm_vals[6], pm_vals[7], pm_vals[8]])
    writer_avg.writerow("")

root_path = "/media/iNicosiaData/engd_data/"
# two files as then the learned_models_benchmarks.py breaks
file_name = sys.argv[1]
file_name_avg = sys.argv[2]

# This file will create two .csv files that contain the results of the binary classifier and their average.
result_file_output_path = os.path.join(root_path, file_name)
result_avg_file_output_path = os.path.join(root_path, file_name_avg)

f = open(result_file_output_path, 'w', encoding='UTF8')
f_avg = open(result_avg_file_output_path, 'w', encoding='UTF8')

writer = csv.writer(f)
writer_avg = csv.writer(f_avg)

write_cmu_res(writer, writer_avg, root_path)
write_retail_shop_res(writer, writer_avg, root_path)
write_lamar_res(writer, writer_avg, root_path)


