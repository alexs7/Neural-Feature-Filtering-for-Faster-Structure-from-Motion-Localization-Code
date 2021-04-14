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
# python3 classification_simple.py colmap_data/Coop_data/slice1/ML_data/ml_database.db 5 16384 500 classification_simple/
# python3 classification_simple.py colmap_data/Coop_data/slice1/ML_data/ml_database.db 5 32768 500 second_simple_classification/

# this might cause problems when loading the model
# def soft_acc(y_true, y_pred):
#     return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

MODEL_NAME = "BinaryClassificationSimple-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="colmap_data/Coop_data/slice1/ML_data/results/{}".format(MODEL_NAME), histogram_freq=1, write_images=True)

print("Running Script..!")
print(MODEL_NAME)

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

sifts_scores = ml_db.execute("SELECT sift, matched FROM data").fetchall() #guarantees same order

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
# min-max normalization
# y_train = ( y_train - y_train.min() ) / ( y_train.max() - y_train.min() )
# y_test = ( y_test - y_test.min() ) / ( y_test.max() - y_test.min() )

print("Saving unseen test data..")
np.save(base_path+"X_test", X_test)
np.save(base_path+"y_test", y_test)

print("y_train mean: " + str(y_train.mean()))
print("y_test mean: " + str(y_test.mean()))

# create model
print("Creating model")
model = Sequential()
# in keras the first layer is a hidden layer too, so input dims is OK here
model.add(Dense(128, input_dim=128, kernel_initializer='normal', activation='sigmoid')) #TODO: relu or sigmoid ?
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

# Compile model
# The loss here will be, binary_crossentropy
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# train
print("Training.. on " + str(X_train.shape[0]) + " samples")
history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2,
                    callbacks=[tensorboard])

import pdb
pdb.set_trace()

print(history.history.keys())
# "Loss"
# The loss here will be, binary_crossentropy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss (binary_crossentropy)', 'Validation Loss (binary_crossentropy)'], loc='upper left')
plt.savefig(base_path + "loss_"+str(fold_no)+".png")

plt.clf()
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('Model Binary Accuracy')
plt.ylabel('Binary Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Binary Accuracy', 'Validation Binary Accuracy'], loc='upper left')
plt.savefig(base_path + "binary_accuracy_"+str(fold_no)+".png")

print("Evaluating..")
acc = model.evaluate(X_train[test], y_train[test], verbose=1, batch_size=batch_size)
print(f"  Binary Accuracy: {acc[1]}")
print(f"  Binary Crossentropy Loss: {acc[0]}")

print("Saving model..")
model.save(base_path+"model")

print("Saving model metrics..")
np.save(base_path + "accuracy", acc[1])

print("Done!")

