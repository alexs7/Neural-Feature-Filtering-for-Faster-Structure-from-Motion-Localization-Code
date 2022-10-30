# This file will load the training data from create_training_data_predicting_matchability.py
# Train a RF classifier according to the paper, and save the model
# to run for all datasets in parallel:
# python3 train_for_predicting_matchability.py colmap_data/CMU_data/slice3 & python3 train_for_predicting_matchability.py colmap_data/CMU_data/slice4 & python3 train_for_predicting_matchability.py colmap_data/CMU_data/slice6 & python3 train_for_predicting_matchability.py colmap_data/CMU_data/slice10 & python3 train_for_predicting_matchability.py colmap_data/CMU_data/slice11 & python3 train_for_predicting_matchability.py colmap_data/Coop_data/slice1 &

import os
import subprocess
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import compute_class_weight
from joblib import dump

base_path = sys.argv[1]
print("Base path: " + base_path)
no_samples = sys.argv[2] #this value is defined from "create_training_data_predicting_matchability.py"

pos_samples_file_name = f"pos_{no_samples}_samples.txt"
neg_samples_file_name = f"neg_{no_samples}_samples.txt"
original_tool_trained_model_output_name = f"rforest_{no_samples}.gz"
sklearn_model_output_name = f"rforest_{no_samples}.joblib"
data_path = os.path.join(base_path, "predicting_matchability_comparison_data")
original_tool_path = "/home/Neural-Feature-Filtering-for-Faster-Structure-from-Motion-Localization-Code/code_to_compare/Predicting_Matchability/rforest"

print("Loading data..(from same location as original tool)")
# data is shuffled
pos_samples_path = os.path.join(original_tool_path, pos_samples_file_name)
positive_samples = np.loadtxt(pos_samples_path)
# data is shuffled
neg_samples_path = os.path.join(original_tool_path, neg_samples_file_name)
negative_samples = np.loadtxt(neg_samples_path)

X_train = np.r_[positive_samples, negative_samples]
y_train = np.r_[np.ones([positive_samples.shape[0]]), np.zeros([negative_samples.shape[0]])]
y_train = y_train.astype(np.uint8)

# random_state, https://stackoverflow.com/questions/39158003/confused-about-random-state-in-decision-tree-of-scikit-learn
# parameters are set from paper Section 3.2
rf = RandomForestClassifier(n_estimators = 25, max_depth = 25, n_jobs=-1)

# 21/11/2022 - Just keeping old weighted code here
# classes = np.array([0,1])
# class_weight = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
# class_weight_dict = dict(zip(classes, class_weight))
# rf_weighted = RandomForestClassifier(n_estimators = 25, max_depth = 25, n_jobs=-1, class_weight = class_weight_dict)

print("Training sklearn model (s)..")
rf.fit(X_train, y_train)

print("Dumping model..")
dump(rf, os.path.join(data_path, sklearn_model_output_name))

print("Training original tool..")

original_rforest_model_path = os.path.join(original_tool_path, original_tool_trained_model_output_name)
original_tool_train_command = os.path.join(original_tool_path, "./rforest")

original_tool_exec = [original_tool_train_command, "-t", "25", "-d", "25", "-p", pos_samples_path, "-n", neg_samples_path, "-f", original_rforest_model_path]
subprocess.check_call(original_tool_exec)

print("Done!")