# This file will load the training data from create_training_data_predicting_matchability.py
# Train a RF classifier according to the paper, and save the model
# to run for all datasets in parallel:
# python3 train_for_predicting_matchability.py colmap_data/CMU_data/slice3 & python3 train_for_predicting_matchability.py colmap_data/CMU_data/slice4 & python3 train_for_predicting_matchability.py colmap_data/CMU_data/slice6 & python3 train_for_predicting_matchability.py colmap_data/CMU_data/slice10 & python3 train_for_predicting_matchability.py colmap_data/CMU_data/slice11 & python3 train_for_predicting_matchability.py colmap_data/Coop_data/slice1

import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data import getTrainingDataForPredictingMatchability
from joblib import dump

base_path = sys.argv[1]
print("Base path: " + base_path)

db_live_path = os.path.join(base_path, "live/database.db")
data_path = os.path.join(base_path, "predicting_matchability_comparison_data")

# random_state, https://stackoverflow.com/questions/39158003/confused-about-random-state-in-decision-tree-of-scikit-learn
# parameters are set from paper Section 3.2
random_samples_no = 485000
rf = RandomForestClassifier(n_estimators = 25, max_depth = 25, random_state = 0)

pos_samples, neg_samples = getTrainingDataForPredictingMatchability(data_path, random_samples_no)
ones = np.ones([pos_samples.shape[0], 1])
zeros = np.zeros([neg_samples.shape[0], 1])
pos_samples = np.c_[pos_samples, ones]
neg_samples = np.c_[neg_samples, zeros]

all_data = np.empty([0,129])
all_data = np.r_[all_data, pos_samples]
all_data = np.r_[all_data, neg_samples]

X = all_data[:,0:128]
y = all_data[:,-1]

print("Training..")
rf.fit(X, y)

print("Dumping model..")
dump(rf, os.path.join(data_path, "rf_model.joblib"))

print("Done!")