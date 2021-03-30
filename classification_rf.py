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
from sklearn.ensemble import RandomForestClassifier

# sample commnad to run on bath cloud servers, ogg .. etc
# python3 classification_rf.py colmap_data/Coop_data/slice1/ML_data/ml_database.db 5 16384 1000 classification_rf_tbd/

# this might cause problems when loading the model
# def soft_acc(y_true, y_pred):
#     return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

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

def binary_accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    '''
    return np.mean(np.equal(y_true, np.round(y_pred)))

print("Running Script..!")

db_path = sys.argv[1]
num_folds = int(sys.argv[2])
batch_size = int(sys.argv[3])
epochs = int(sys.argv[4])
model_base_name = sys.argv[5]
base_path = "colmap_data/Coop_data/slice1/ML_data/results/"
os.makedirs(base_path+model_base_name)
base_path = base_path+model_base_name

print("num_folds: " + str(num_folds))
print("batch_size: " + str(batch_size))
print("epochs: " + str(epochs))

ml_db = COLMAPDatabase.connect_ML_db(db_path)

sifts_scores = ml_db.execute("SELECT * FROM (SELECT sift, matched FROM data WHERE matched = 1 LIMIT 1000000) UNION SELECT * FROM (SELECT sift, matched FROM data WHERE matched = 0 LIMIT 1000000)").fetchall() #guarantees same order

all_sifts = (COLMAPDatabase.blob_to_array(row[0] , np.uint8) for row in sifts_scores)
all_sifts = np.array(list(all_sifts))

all_targets = (row[1] for row in sifts_scores) #binary values
all_targets = np.array(list(all_targets))

print("Splitting data into test/train..")

# X_train, y_train, X_test, y_test = split_data(all_sifts, all_targets, 0.3, randomize = True)
X_train, X_test, y_train, y_test = train_test_split(all_sifts, all_targets, test_size=0.1, shuffle=True, random_state=42)

print("Total Training Size: " + str(X_train.shape[0]))
print("Total Test Size: " + str(X_test.shape[0]))

# standard scaling - mean normalization
X_train = ( X_train - X_train.mean() ) / X_train.std()
X_test = ( X_test - X_test.mean() ) / X_test.std()

print("Saving unseen test data..")
np.save(base_path+"X_test", X_test)
np.save(base_path+"y_test", y_test)

print("y_train mean: " + str(y_train.mean()))
print("y_test mean: " + str(y_test.mean()))

eval_scores = []
kfold = KFold(n_splits = num_folds, shuffle = True, random_state=42)
fold_no = 1
for train, test in kfold.split(X_train):
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # create model
    print("Creating model")
    rf = RandomForestClassifier(n_estimators = 1000,
                               random_state=42, max_depth = 40,
                               n_jobs = -1)

    # train
    print("Training.. on " + str(X_train[train].shape[0]) + " samples")
    rf.fit(X_train[train], y_train[train])

    print("Evaluating..")
    predictions = rf.predict(X_train[test])

    acc = binary_accuracy(y_train[test], predictions)
    print(f"  Binary Accuracy: {acc}")
    # save only binary accuracy for now ?
    eval_scores.append(acc)

    print("Saving model..")
    dump(rf, base_path+"model_"+str(fold_no)+".joblib")
    fold_no +=1

    print("Saving model metrics..")
    np.save(base_path + "acc_" + str(fold_no), acc)
    fold_no +=1

accu_mean = np.mean(eval_scores)
print(" -> Total Accuraccy Mean: " + str(accu_mean))
np.save(base_path+"acc_mean", accu_mean)
print("Done!")

