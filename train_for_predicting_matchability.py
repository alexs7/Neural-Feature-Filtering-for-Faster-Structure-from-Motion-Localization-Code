# This file will load the training data from create_training_data_predicting_matchability.py
# Train a RF classifier according to the paper, and save the model
# to run for all datasets in parallel:
# python3 train_for_predicting_matchability.py colmap_data/CMU_data/slice3 & python3 train_for_predicting_matchability.py colmap_data/CMU_data/slice4 & python3 train_for_predicting_matchability.py colmap_data/CMU_data/slice6 & python3 train_for_predicting_matchability.py colmap_data/CMU_data/slice10 & python3 train_for_predicting_matchability.py colmap_data/CMU_data/slice11 & python3 train_for_predicting_matchability.py colmap_data/Coop_data/slice1 &

import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data import getClassificationData
from joblib import dump
from sklearn.metrics import f1_score

base_path = sys.argv[1]
print("Base path: " + base_path)

db_live_path = os.path.join(base_path, "live/database.db")
data_path = os.path.join(base_path, "predicting_matchability_comparison_data")
db_pm =  os.path.join(data_path, "training_data.db")

# random_state, https://stackoverflow.com/questions/39158003/confused-about-random-state-in-decision-tree-of-scikit-learn
# parameters are set from paper Section 3.2, but I train on all data.
rf = RandomForestClassifier(n_estimators = 25, max_depth = 25, random_state = 0, n_jobs=-1)

print("Loading data..")
# using same methods I used for my NNs. as the database structure is the same
# data is shuffled
sift_vecs, classes = getClassificationData(db_pm)
classes = classes.astype(np.uint8)

validation_size= 15000
X_train = sift_vecs[0:(sift_vecs.shape[0] - validation_size),:]
y_train = classes[0:(classes.shape[0] - validation_size)]

X_val = sift_vecs[-validation_size:,:]
y_val = classes[-validation_size:]

print("Training..")
rf.fit(X_train, y_train)

print(f"f1_score 0-1 : {f1_score(y_val, rf.predict(X_val))}")

print("Dumping model..")
dump(rf, os.path.join(data_path, "rf_model.joblib"))

print("Done!")