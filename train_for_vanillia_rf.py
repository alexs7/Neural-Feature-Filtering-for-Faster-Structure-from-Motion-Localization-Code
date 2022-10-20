# This file will load the training data for my original Neural Networks.
# This will be used to compare the models from Predicting Matchability and Match no Match but with my own data this time.
# This means tos use their models but my data. My data uses 3D points averages descriptors, not matches between images - remember ?

import os
import sys
from sklearn.ensemble import RandomForestClassifier
from data import getClassificationData
from joblib import dump

base_path = sys.argv[1]

print("Base path: " + base_path)

# TODO Continue from here
output_path = os.path.join(base_path, "ml_models_vanillia_comparison_data")
os.makedirs(output_path, exist_ok = True)
ml_db = os.path.join(base_path, "ML_data/ml_database_all.db")

print("Now train a vanillia RF -just to use as a baseline")
sift_vecs, classes = getClassificationData(ml_db)

# comparison models
rf_pm = RandomForestClassifier(n_estimators = 25, max_depth = 25, random_state = 0)
rf_mnm = RandomForestClassifier(n_estimators = 5, max_depth = 5, min_samples_split = 2, n_jobs=-1)

print("Training PM..")
rf_pm.fit(sift_vecs, classes)
rf_mnm.fit(sift_vecs, classes)

print("Dumping models..")
dump(rf_pm, os.path.join(output_path, "rf_model_vanillia_pm.joblib"))
dump(rf_mnm, os.path.join(output_path, "rf_model_vanillia_mnm.joblib"))

print("Done!")