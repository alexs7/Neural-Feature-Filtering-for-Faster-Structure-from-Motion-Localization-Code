import os
import time
import matplotlib.pyplot as plt
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
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

# sample commnad to run on bath cloud servers, ogg .. etc
# python3 classification.py colmap_data/Coop_data/slice1/ML_data/ml_database_class.db 16384 500

MODEL_NAME = "BinaryClassificationSimple-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="colmap_data/Coop_data/slice1/ML_data/results/{}".format(MODEL_NAME), histogram_freq=1, write_images=True)

print("Running Script..!")
print(MODEL_NAME)

db_path = sys.argv[1]
batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])

print("Batch_size: " + str(batch_size))
print("Epochs: " + str(epochs))

print("Loading data..")
ml_db = COLMAPDatabase.connect_ML_db(db_path)

data = ml_db.execute("SELECT sift, matched FROM data").fetchall() #guarantees same order

sift_vecs = (COLMAPDatabase.blob_to_array(row[0] , np.uint8) for row in data)
sift_vecs = np.array(list(sift_vecs))

classes = (row[1] for row in data) #binary values
classes = np.array(list(classes))

print("Total Training Size: " + str(sift_vecs.shape[0]))

# standard scaling - mean normalization
sift_vecs = ( sift_vecs - sift_vecs.mean() ) / sift_vecs.std()

# min-max normalization classes - maybe not needed for binary classification ?

import pdb
pdb.set_trace()

# Create model
print("Creating model")
model = Sequential()
# in keras the first layer is a hidden layer too, so input dims is OK here
model.add(Dense(128, input_dim=128, kernel_initializer='normal', activation='sigmoid')) #TODO: relu or sigmoid ?
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model
# The loss here will be, binary_crossentropy
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# Train
history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2,
                    callbacks=[tensorboard])

print("Done!")

