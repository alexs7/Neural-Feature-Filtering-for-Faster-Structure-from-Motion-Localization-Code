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

# OpenCV model
# NOTE (31/08/2022) The OpenCV RF model has problem with predict() - it just returns zeros:
priors = np.array([0.4, 0.6]).reshape([1,2])
rtree = cv2.ml.RTrees_create()
rtree.setMinSampleCount(2)
rtree.setRegressionAccuracy(0)
rtree.setUseSurrogates(False)
rtree.setPriors(priors)
rtree.setCalculateVarImportance(True)
rtree.setActiveVarCount(2)
rtree.setMaxDepth(5)
rtree.setTermCriteria(( cv2.TERM_CRITERIA_MAX_ITER, 5, 0 ))

rdata = getTrainingDataForMatchNoMatch(data_path)

X = rdata[:,:133].astype(np.float32)
# https://stackoverflow.com/questions/36440266/how-to-use-opencv-rtrees-for-binary-classification
y = rdata[:,133].astype(np.int32) # this needs to be int32 for classification

# https://stackoverflow.com/questions/53181119/python-opencv-rtrees-does-not-load-properly
train_data = cv2.ml.TrainData_create(samples=X, layout=cv2.ml.ROW_SAMPLE, responses=y)

print("Training..(OpenCV model)")
rtree.train(trainData=train_data)

err = rtree.calcError(train_data, True)[0]

print("Dumping model..")
rtree.save(os.path.join(data_path, "rf_match_no_match_opencv.xml"))
np.savetxt(os.path.join(data_path, "rf_generalization_error.txt"), [err])

# SkLearn Model
rf = RandomForestClassifier(n_estimators = 5, max_depth = 5, random_state = 0, class_weight={0:0.4, 1:0.6})

print("Training..")
rf.fit(X, y)

print("Dumping model..")
dump(rf, os.path.join(data_path, "rf_match_no_match_sk.joblib"))

print("Done!")