# This file will load all models from test_match_no_match_tool.py and test_predicting_matchability_tool.py and my NN classifier and
# generate the confusion matrix recall, and precision number, etc csv file
# The difference from test_match_no_match_tool.py is that I test on my own 3D data which makes more sense. So the test_match_no_match_tool.py is not used anymore
# There is not much difference with test_predicting_matchability_tool.py (training on same data) but I use the python model here only (which performs the same as the C++ tool).
# Here all the models are test at once so easier to examine results.

# A minor note here is that you can use the GT data definition from the papers to evaluate the models from your paper but,
# when it comes to benchmarking for your thesis
# you will have to use the 3D data, a SIFT feature <-> matched or not to a 3D point
# But! because you use matches from the COLMAP two views geometry table to train both models from the papers it is fine because
# each match in that table is used in the 3D recostruction - so it has most probably a 3D point matched to it.

# After you are happy with the results from here you can move on to the model_evaluator*.py files
# Use this file to pick the best performing model for the model_evaluator scripts.

import csv
import os
import sys
import cv2
import numpy as np
from joblib import load
from sklearn.metrics import classification_report, precision_recall_fscore_support, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, balanced_accuracy_score
from generate_gt_truth_testing import createGTTestDataForPM, createGTTestDataForMnM
from tensorflow import keras

def write_test_to_file(test_data, original_too_test_data_path):
    print(f"Test data size: {test_data.shape[0]}")
    with open(original_too_test_data_path, 'w') as f:
        for desc in test_data:
            row = ' '.join([str(num) for num in desc[0:128].astype(np.uint8)])
            f.write(f"{row}\n")
    print("Test data saved!")

def load_predictions(original_output_results_path):
    res = np.loadtxt(original_output_results_path)
    y_pred = np.empty([res.shape[0]])
    for i in range(res.shape[0]):
        el = res[i]
        pos = el[1]
        if (pos >= 0.5):
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred

base_path = sys.argv[1]
mnm_base_path = sys.argv[2]
nn_model_path = sys.argv[3] # i.e. "colmap_data/tensorboard_results/classification_Extended_CMU_slice3/early_stop_model/"

no_samples_pm = [500, 1000, 1500, 2000] #these are in my notebook, I decided to use these values
no_samples_mnm = [2000, 4000, 8000, 16000] #these are in my notebook, I decided to use these values
ml_path = os.path.join(base_path, "ML_data")
result_file_output_path = os.path.join(ml_path, "binary_models_metrics.csv")

# Start writing in csv file
header = ['Model samples size', 'Precision (Positive Class)', 'Recall (Positive Class)', 'F1 Score', 'TN', 'FN', 'FP', 'TP', 'Balanced Accuracy']
with open(result_file_output_path, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    print("Getting PM & NN test data..")
    test_data_pm = createGTTestDataForPM(base_path)  # N x 129
    X_test = test_data_pm[:, 0:128]
    y_true = test_data_pm[:, 128].astype(np.uint8)  # or y_test

    writer.writerow(["PM (2014)", " ", " ", " ", " ", " ", " ", " ", " "])
    print("Loading PM python models..")
    for no_samples in no_samples_pm:
        print(f"{no_samples} samples --->")
        sklearn_model_path = os.path.join(os.path.join(base_path, "predicting_matchability_comparison_data"), f"rforest_{no_samples}.joblib")
        rf = load(sklearn_model_path)
        y_pred = rf.predict(X_test)

        print(classification_report(y_true, y_pred, labels=[0, 1]))
        print(precision_recall_fscore_support(y_true, y_pred, labels=[0, 1]))
        p_score = precision_score(y_true, y_pred)
        r_score = recall_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        f1_s = f1_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        tp_perc = tp * 100 / len(X_test)
        tn_perc = tn * 100 / len(X_test)
        fn_perc = fn * 100 / len(X_test)
        fp_perc = fp * 100 / len(X_test)

        writer.writerow([no_samples, p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc])
        print("--->")

    # spacer
    writer.writerow([" ", " ", " ", " ", " ", " ", " ", " ", " "])

    writer.writerow(["MnM (2020)", " ", " ", " ", " ", " ", " ", " ", " "])

    print("Getting MnM test data..")
    test_data_mnm = createGTTestDataForMnM(base_path, mnm_base_path)  # N x 129
    X_test = test_data_mnm[:, 0:8]
    y_true = test_data_mnm[:, 8].astype(np.uint8)  # or y_test

    print("Loading MnM C++ models using python..")
    for no_samples in no_samples_mnm:
        print(f"{no_samples} samples --->")
        model_path = os.path.join(base_path, "match_or_no_match_comparison_data", f"Trained model {no_samples}.xml")
        cpp_original_model = cv2.ml.RTrees_load(model_path)
        _, y_pred = cpp_original_model.predict(X_test)

        print(classification_report(y_true, y_pred, labels=[0, 1]))
        print(precision_recall_fscore_support(y_true, y_pred, labels=[0, 1]))
        p_score = precision_score(y_true, y_pred)
        r_score = recall_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        f1_s = f1_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        tp_perc = tp * 100 / len(X_test)
        tn_perc = tn * 100 / len(X_test)
        fn_perc = fn * 100 / len(X_test)
        fp_perc = fp * 100 / len(X_test)

        writer.writerow([no_samples, p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc])
        print("--->")

    # no samples here, just 1 NN
    writer.writerow(["NNs", " ", " ", " ", " ", " ", " ", " ", " "])

    print("Loading my NN models..")
    classification_model = keras.models.load_model(nn_model_path)
    # PM data here can also be used for my NN as they were trained on the same data
    X_test = test_data_pm[:, 0:128]
    y_true = test_data_pm[:, 128].astype(np.uint8)  # or y_test

    classifier_predictions = classification_model.predict_on_batch(X_test)
    matchable_desc_indices = np.where(classifier_predictions > 0.5)[0]
    predictions = np.zeros(len(classifier_predictions))
    predictions[matchable_desc_indices] = 1
    y_pred = predictions.astype(np.uint8)

    print(classification_report(y_true, y_pred, labels=[0, 1]))
    print(precision_recall_fscore_support(y_true, y_pred, labels=[0, 1]))
    p_score = precision_score(y_true, y_pred)
    r_score = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1_s = f1_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    tp_perc = tp * 100 / len(X_test)
    tn_perc = tn * 100 / len(X_test)
    fn_perc = fn * 100 / len(X_test)
    fp_perc = fp * 100 / len(X_test)

    writer.writerow(["All", p_score, r_score, f1_s, tn_perc, fn_perc, fp_perc, tp_perc, bal_acc])

    print("--->")

print("Done!")