# This file will load the training data for my original Neural Networks.
# This will be used to compare the models from Predicting Matchability and Match no Match but with my own data this time.
# This means to use their models but my data. My data uses 3D points averages descriptors, not matches between images - remember ?

import os
import sys
from sklearn.ensemble import RandomForestClassifier
from data import getClassificationData
from joblib import dump

base_path = sys.argv[1]

print("Base path: " + base_path)
ml_db = os.path.join(base_path, "ML_data/ml_database_all.db")

comparison_data_path_PM_trained_my_data = os.path.join(base_path, "predicting_matchability_comparison_data_trained_on_my_data")
comparison_data_path_MoNM_trained_my_data = os.path.join(base_path, "match_or_no_match_comparison_data_trained_on_my_data")

os.makedirs(comparison_data_path_PM_trained_my_data, exist_ok=True)
os.makedirs(comparison_data_path_MoNM_trained_my_data, exist_ok=True)

print("Getting data..")
sift_vecs, classes = getClassificationData(ml_db)

# comparison models
rf_pm = RandomForestClassifier(n_estimators = 25, max_depth = 25, random_state = 0)
rf_mnm = RandomForestClassifier(n_estimators = 5, max_depth = 5, min_samples_split = 2, n_jobs=-1)

print("Training PM..")
rf_pm.fit(sift_vecs, classes)

print("Training MnM..")
rf_mnm.fit(sift_vecs, classes)

print("Dumping models..")
dump(rf_pm, os.path.join(comparison_data_path_PM_trained_my_data, "rf_model.joblib"))
dump(rf_mnm, os.path.join(comparison_data_path_MoNM_trained_my_data, "rf_model.joblib"))

print("Done!")