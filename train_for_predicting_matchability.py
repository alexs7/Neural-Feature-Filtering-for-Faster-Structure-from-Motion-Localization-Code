# This file will load the training data from create_training_data_predicting_matchability.py
# Train a RF classifier (sklearn and original c++ tool) according to the paper, and save the model

# TODO: 22/12/2022 comment out the original tool since I am running this on the CYENS machine (the original tool is on the weatherwax)

import os
import subprocess
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data import getClassificationDataPM
from joblib import dump

from database import COLMAPDatabase
from parameters import Parameters


def train_and_save_model(parameters, no_samples=1200):
    print("Training on OpenCV SIFT data..")
    training_data_db_path = os.path.join(parameters.predicting_matchability_comparison_data_full, f"training_data_{no_samples}_samples_opencv.db")

    print("Loading training data from: " + training_data_db_path)
    sift, classes = getClassificationDataPM(training_data_db_path)

    # random_state, https://stackoverflow.com/questions/39158003/confused-about-random-state-in-decision-tree-of-scikit-learn
    # parameters are set from paper Section 3.2
    rf = RandomForestClassifier(n_estimators = 25, max_depth = 25, n_jobs=-1)

    print("Training sklearn model (s)..")
    # just for readability
    X_train = sift
    y_train = classes
    rf.fit(X_train, y_train)

    print("Dumping model..")
    sklearn_model_output_name = f"rforest_{no_samples}.joblib"
    dump(rf, os.path.join(parameters.predicting_matchability_comparison_data_full, sklearn_model_output_name))

root_path = "/media/iNicosiaData/engd_data/"
dataset = sys.argv[1]

# Note: Use 3200 max_pairs for HGE, CAB, LIN and 1200 for CMU

if(dataset == "HGE" or dataset == "CAB" or dataset == "LIN"):
    base_path = os.path.join(root_path, "lamar", f"{dataset}_colmap_model")
    print("Base path: " + base_path)
    params = Parameters(base_path)
    train_and_save_model(params, no_samples=params.predicting_matchability_comparison_data_lamar_no_samples)

if(dataset == "CMU"):
    slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                    "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    for slice_name in slices_names:
        base_path = os.path.join(root_path, "cmu", f"{slice_name}", "exmaps_data")
        print("Base path: " + base_path)
        params = Parameters(base_path)
        train_and_save_model(params)

if(dataset == "RetailShop"):
    base_path = os.path.join(root_path, "retail_shop", "slice1")
    print("Base path: " + base_path)
    params = Parameters(base_path)
    train_and_save_model(params)

print("Done!")