# This file will load the training data from create_training_data_predicting_matchability.py
# Train a RF classifier according to the paper, and save the model
# to run for all datasets in parallel:
# python3 train_for_match_no_match.py colmap_data/CMU_data/slice3 & python3  train_for_match_no_match.py colmap_data/CMU_data/slice4 & python3  train_for_match_no_match.py colmap_data/CMU_data/slice6 & python3  train_for_match_no_match.py colmap_data/CMU_data/slice10 & python3  train_for_match_no_match.py colmap_data/CMU_data/slice11 & python3  train_for_match_no_match.py colmap_data/Coop_data/slice1 &

import os
import sys
import numpy as np
from data import getTrainingDataForMatchNoMatch
import cv2
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

base_path = sys.argv[1]
print("Base path: " + base_path)

data_path = os.path.join(base_path, "match_or_no_match_comparison_data")
rdata = getTrainingDataForMatchNoMatch(data_path)

# SkLearn Model
rf = RandomForestClassifier(n_estimators = 5, max_depth = 5, min_samples_split = 2, n_jobs=-1)
# [xs, ys, octaves, angles, sizes, responses, dominantOrientations, green_intensities, matcheds]
X = rdata[:,0:8]
y = rdata[:,8]

print("Training 5 by 5 RF.. MnM Paper")
rf.fit(X, y)

print("Dumping model (s)..")
dump(rf, os.path.join(data_path, "rf_model.joblib"))

print("Done!")

# OpenCV model
# NOTE (31/08/2022) The OpenCV RF model has problem with predict() - it just returns zeros:
# priors = np.array([0.4, 0.6]).reshape([1,2])
# rtree = cv2.ml.RTrees_create()
# rtree.setMinSampleCount(2)
# rtree.setRegressionAccuracy(0)
# rtree.setUseSurrogates(False)
# rtree.setPriors(priors)
# rtree.setCalculateVarImportance(True)
# rtree.setActiveVarCount(2)
# rtree.setMaxDepth(5)
# rtree.setTermCriteria(( cv2.TERM_CRITERIA_MAX_ITER, 5, 0 ))

# old opencv model from paper - not used
# # https://stackoverflow.com/questions/53181119/python-opencv-rtrees-does-not-load-properly
# train_data = cv2.ml.TrainData_create(samples=X, layout=cv2.ml.ROW_SAMPLE, responses=y)
#
# print("Training..(OpenCV model)")
# rtree.train(trainData=train_data)
#
# err = rtree.calcError(train_data, True)[0]
#
# print("Dumping model..")
# rtree.save(os.path.join(data_path, "rf_match_no_match_opencv.xml"))
# np.savetxt(os.path.join(data_path, "rf_generalization_error.txt"), [err])