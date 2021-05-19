import os
import time
import matplotlib.pyplot as plt
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard
import sys
import glob
import pandas as pd
from database import COLMAPDatabase
# from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import train_test_split
from custom_callback import CustomCallback
from sklearn.preprocessing import StandardScaler

metrics = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='binary_accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

# sample commnad to run on bath cloud servers, ogg .. etc
# python3 classification_2.py colmap_data/Coop_data/slice1/ML_data/ml_database_train.db 32768 800 ManyLayers

db_path = sys.argv[1]
batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])
name = sys.argv[4]

MODEL_NAME = "BinaryClassification-{}-{}".format( name, time.ctime())
log_dir = "colmap_data/Coop_data/slice1/ML_data/results/{}".format(MODEL_NAME)
cust_log_dir = "colmap_data/Coop_data/slice1/ML_data/results/{}".format(MODEL_NAME) + "/my_metrics"

print("TensorBoard log_dir: " + log_dir)
print("CustomCallback log_dir: " + cust_log_dir)

tensorboard_cb = TensorBoard(log_dir=log_dir)
# cust_callback = CustomCallback(cust_log_dir)

all_callbacks = [tensorboard_cb] #removed cust_callback for now

print("Running Script..!")
print(MODEL_NAME)

print("Batch_size: " + str(batch_size))
print("Epochs: " + str(epochs))

print("Loading data..")
ml_db = COLMAPDatabase.connect_ML_db(db_path)

data = ml_db.execute("SELECT sift, matched FROM data").fetchall() #guarantees same order - maybe ?

sift_vecs = (COLMAPDatabase.blob_to_array(row[0] , np.uint8) for row in data)
sift_vecs = np.array(list(sift_vecs))

classes = (row[1] for row in data) #binary values
classes = np.array(list(classes))

ratio = np.where(classes == 1)[0].shape[0] / np.where(classes == 0)[0].shape[0]
print("Ratio of Positives to Negatives: " + str(ratio))

print("Total Training Size: " + str(sift_vecs.shape[0]))

# standard scaling - mean normalization
# scaler = StandardScaler()
# sift_vecs = scaler.fit_transform(sift_vecs)

# Create model
print("Creating model")
model = Sequential()
# in keras the first layer is a hidden layer too, so input dims is OK here
model.add(Dense(128, input_dim=128, activation='relu')) #TODO: relu or sigmoid ?
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
opt = keras.optimizers.Adam(learning_rate=3e-4)
# The loss here will be, binary_crossentropy
model.compile(optimizer=opt, loss=keras.losses.BinaryCrossentropy(), metrics=metrics)
model.summary()

# Before training you should use a baseline model

# Train (or fit() )
# Just for naming's sake
X_train = sift_vecs
y_train = classes
history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=epochs,
                    shuffle=True,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=all_callbacks)

# Save model here
print("Saving model")
model_save_path = os.path.join("colmap_data/Coop_data/slice1/ML_data/results/", MODEL_NAME, "model")
model.save(model_save_path) #double check with keras.models.load_model(model_save_path) in debug

import pdb
pdb.set_trace()

print("Done!")

