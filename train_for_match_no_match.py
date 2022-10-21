# This file will load the training data from create_training_data_predicting_matchability.py
# Train a RF classifier according to the paper, and save the model
# to run for all datasets in parallel:
# python3 train_for_match_no_match.py colmap_data/CMU_data/slice3 & python3  train_for_match_no_match.py colmap_data/CMU_data/slice4 & python3  train_for_match_no_match.py colmap_data/CMU_data/slice6 & python3  train_for_match_no_match.py colmap_data/CMU_data/slice10 & python3  train_for_match_no_match.py colmap_data/CMU_data/slice11 & python3  train_for_match_no_match.py colmap_data/Coop_data/slice1 &

import os
import sys
import numpy as np
from sklearn.metrics import f1_score
from data import getTrainingAndTestDataForMatchNoMatch
import cv2
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

from database import COLMAPDatabase

base_path = sys.argv[1]
print("Base path: " + base_path)

data_path = os.path.join(base_path, "match_or_no_match_comparison_data")
all_data = getTrainingAndTestDataForMatchNoMatch(data_path)

isTestSampleIdx = 9
train_data = all_data[np.where(all_data[:, isTestSampleIdx] == 0)[0]]
test_data = all_data[np.where(all_data[:, isTestSampleIdx] == 1)[0]]

# [xs, ys, octaves, angles, sizes, responses, dominantOrientations, green_intensities, matcheds]
X_train = train_data[:, 0:8]
y_train = train_data[:, 8].astype(np.uint8)

print(f"Training Data Size: {X_train.shape[0]}")

validation_size= 10000
X_train = X_train[0:(X_train.shape[0] - validation_size),:]
y_train = y_train[0:(y_train.shape[0] - validation_size)]

X_val = X_train[-validation_size:,:]
y_val = y_train[-validation_size:].astype(np.uint8)

# TODO: this might change for diff datasets! So set it appropriately
weight = int(np.where( y_train == 0 )[0].shape[0] / np.where( y_train == 1 )[0].shape[0])

# SkLearn Model
rf = RandomForestClassifier(n_estimators = 5, max_depth = 5, random_state = 0,
                            n_jobs=-1, class_weight={0:1,1:weight})

print("Training RF.. MnM Paper")
rf.fit(X_train, y_train)

print(f"val. f1_score 0-1 : {f1_score(y_val, rf.predict(X_val))}")

print("Testing RF.. MnM Paper")

X_test = test_data[:, 0:8]
y_test = test_data[:, 8].astype(np.uint8)

print(f"test f1_score 0-1 : {f1_score(y_test, rf.predict(X_test))}")

print("Dumping model (s)..")
dump(rf, os.path.join(data_path, "rf_model.joblib"))

print("Done!")


# debug code below
#
# def get_keypoints_data(db, img_id, image_file):
#     # it is a row, with many keypoints (blob)
#     kp_db_row = db.execute("SELECT rows, cols, data, dominantOrientations FROM keypoints WHERE image_id = " + "'" + str(img_id) + "'").fetchone()
#     cols = kp_db_row[1]
#     rows = kp_db_row[0]
#     # x, y, octave, angle, size, response
#     kp_data = COLMAPDatabase.blob_to_array(kp_db_row[2], np.float32)
#     kp_data = kp_data.reshape([rows, cols])
#     xs = kp_data[:,0]
#     ys = kp_data[:,1]
#     dominantOrientations = COLMAPDatabase.blob_to_array(kp_db_row[3], np.uint8)
#     dominantOrientations = dominantOrientations.reshape([rows, 1])
#     indxs = np.c_[np.round(ys), np.round(xs)].astype(np.int)
#     greenInt = image_file[(indxs[:, 0], indxs[:, 1])][:, 1]
#
#     # xs, ys, octaves, angles, sizes, responses, dominantOrientations, greenInt
#     return np.c_[kp_data, dominantOrientations, greenInt]
#
# db_gt_mnm_path = os.path.join("colmap_data/CMU_data/slice3_mnm", "gt/database.db")
# db = COLMAPDatabase.connect(db_gt_mnm_path)
# image_id = '2511'
# image_gt_path = os.path.join('colmap_data/CMU_data/slice3_mnm/gt/images/', 'session_7/img_00980_c0_1288792399837266us.jpg')
# query_image_file = cv2.imread(image_gt_path)
# keypoints_data = get_keypoints_data(db, image_id, query_image_file)

# np.where(test_data[:,10] == int(image_id))
#
# rf.predict(keypoints_data)
#
# f1_score(y_test[0:744], rf.predict(X_test[0:744]))

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