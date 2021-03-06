import os
import time
import matplotlib.pyplot as plt
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
import sys
import glob
import pandas as pd
from database import COLMAPDatabase
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import train_test_split
from joblib import dump, load

# This file contains a RF implementation

# sample command to run on bath cloud servers, ogg .. etc
# python3 regression_rf.py colmap_data/Coop_data/slice1/ML_data/ml_database.db 5 1000 first_rf/
# python3 regression_rf.py colmap_data/Coop_data/slice1/ML_data/ml_database.db 5 100 second_rf/

def split_data(features, target, test_percentage, randomize = False):
    if(randomize):
        print("Randomizing data")
        union = np.c_[features, target]
        np.random.shuffle(union)
        features = union[:, 0:128]
        target = union[:, 128]
    rows_no = features.shape[0] #or test , same thing
    train_percentage = 1 - test_percentage
    train_max_idx = int(np.floor(rows_no * train_percentage))
    X_train = features[0 :  train_max_idx , :]
    y_train = target[0 : train_max_idx]
    X_test = features[train_max_idx : , :]
    y_test = target[train_max_idx :]
    return X_train, y_train, X_test, y_test

print("Running Script (RF)..!")

db_path = sys.argv[1]
num_folds = int(sys.argv[2])
num_of_trees = int(sys.argv[3])
model_base_name = sys.argv[4]
base_path = "colmap_data/Coop_data/slice1/ML_data/results/"
os.makedirs(base_path+model_base_name)
base_path = base_path+model_base_name

print("num_folds: " + str(num_folds))
print("num_of_trees: " + str(num_of_trees))

ml_db = COLMAPDatabase.connect_ML_db(db_path)

sifts = ml_db.execute("SELECT sift FROM data").fetchall()
scores = ml_db.execute("SELECT score FROM data").fetchall()

all_sifts = (COLMAPDatabase.blob_to_array(sift[0] ,np.uint8) for sift in sifts)
all_sifts = np.array(list(all_sifts))

all_scores = (score[0] for score in scores)
all_scores = np.array(list(all_scores))

print("Splitting data into test/train..")

# X_train, y_train, X_test, y_test = split_data(all_sifts, all_scores, 0.3, randomize = True)
X_train, X_test, y_train, y_test = train_test_split(all_sifts, all_scores, test_size=0.2, shuffle=True, random_state=42)

print("Total Training Size: " + str(X_train.shape[0]))
print("Total Test Size: " + str(X_test.shape[0]))

# standard scaling - mean normalization
X_train = ( X_train - X_train.mean() ) / X_train.std()
X_test = ( X_test - X_test.mean() ) / X_test.std()
# min-max normalization
y_train = ( y_train - y_train.min() ) / ( y_train.max() - y_train.min() )
y_test = ( y_test - y_test.min() ) / ( y_test.max() - y_test.min() )

print("Saving unseen test data..")
np.save(base_path+"X_test", X_test)
np.save(base_path+"y_test", y_test)

print("y_train mean: " + str(y_train.mean()))
print("y_test mean: " + str(y_test.mean()))

eval_scores = []
mse_scores_test = []
kfold = KFold(n_splits = num_folds, shuffle = True, random_state=42)
fold_no = 1
for train, test in kfold.split(X_train):
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # create model
    print("Creating model")
    rf = RandomForestRegressor(n_estimatorsint = num_of_trees, random_state=42)

    # train
    print("Training.. on " + str(X_train[train].shape[0]) + " samples")
    rf.fit(X_train[train], y_train[train])

    print("Predicting")
    predictions = rf.predict(X_train[test])
    mse_test_val = metrics.mean_squared_error(y_train[test], predictions)
    print("MSE on Testing Data (K-Fold): " + str(mse_test_val))
    mse_scores_test.append(mse_test_val)

    print("Saving model..")
    dump(clf, base_path+'model.joblib')
    fold_no +=1

mse_mean = np.mean(mse_scores_test)

print("MSE mean: " + str(mse_mean))

np.save(base_path+"mse_mean", mse_mean)

print("Done!")

# print("Evaluate Model..")
# model.evaluate(X_test, y_test, verbose=2)
#
# print("y_train mean: " + str(y_train.mean()))
# print("y_test mean: " + str(y_test.mean()))
#
# print("Saving Model")
# model.save(db_path.rsplit('/', 1)[0]+"/model")

# evaluation
# estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=20, verbose=0)
# kfold = KFold(n_splits=10)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
