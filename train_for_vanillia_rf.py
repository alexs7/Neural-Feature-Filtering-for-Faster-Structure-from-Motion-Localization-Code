# This file will load the training data for my original Neural Networks.
# Train a vanillia RF classifier according
# to run for all datasets in parallel:
# python3 train_for_vanillia_rf.py colmap_data/CMU_data/slice3 25 25 & python3 train_for_vanillia_rf.py colmap_data/CMU_data/slice4 25 25 & python3 train_for_vanillia_rf.py colmap_data/CMU_data/slice6 25 25 & python3 train_for_vanillia_rf.py colmap_data/CMU_data/slice10 25 25 & python3 train_for_vanillia_rf.py colmap_data/CMU_data/slice11 25 25 & python3 train_for_vanillia_rf.py colmap_data/Coop_data/slice1 25 25 &

import os
import sys
from sklearn.ensemble import RandomForestClassifier
from data import getClassificationData
from joblib import dump

base_path = sys.argv[1]
n_estimators = int(sys.argv[2])
max_depth = int(sys.argv[3])

print("Base path: " + base_path)

output_path = os.path.join(base_path, "ml_models_vanillia_comparison_data")
os.makedirs(output_path, exist_ok = True)
ml_db = os.path.join(base_path, "ML_data/ml_database_all.db")

print("Now train a vanillia RF -just to use as a baseline")
rf_vanillia = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state = 0)
sift_vecs, classes = getClassificationData(ml_db)

print("Training..")
rf_vanillia.fit(sift_vecs, classes)

print("Dumping model..")
dump(rf_vanillia, os.path.join(output_path, f"rf_model_vanillia_{n_estimators}_{max_depth}.joblib"))

print("Done!")