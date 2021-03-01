import os
import time
import matplotlib.pyplot as plt
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as K
import sys
import glob
import pandas as pd
from database import COLMAPDatabase

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

db_path = sys.argv[1]
ml_db = COLMAPDatabase.connect_ML_db(db_path)

sifts = ml_db.execute("SELECT sift FROM data").fetchall()
scores = ml_db.execute("SELECT score FROM data").fetchall()

all_sifts = (COLMAPDatabase.blob_to_array(sift[0] ,np.uint8) for sift in sifts)
all_sifts = np.array(list(all_sifts))

all_scores = (score[0] for score in scores)
all_scores = np.array(list(all_scores))

print("Splitting data into test/train..")
X_train, y_train, X_test, y_test = split_data(all_sifts, all_scores, 0.3, randomize = True)

# standard scaling - mean normalization
X_train = ( X_train - X_train.mean() ) / X_train.std()
X_test = ( X_test - X_test.mean() ) / X_test.std()
# min-max normalization
y_train = ( y_train - y_train.min() ) / ( y_train.max() - y_train.min() )
y_test = ( y_test - y_test.min() ) / ( y_test.max() - y_test.min() )

# create model
print("Creating model")
model = Sequential()
# in keras the first layer is a hidden layer too, so input dims is OK here
model.add(Dense(128, input_dim=128, kernel_initializer='normal', activation='relu')) #I know all input values will be positive at this point (SIFT)
model.add(Dense(64, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, kernel_initializer='normal', activation='relu'))
model.add(Dense(16, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid')) #TODO: relu might be more appropriate here (since score can never be negative)

# Compile model
model.compile(optimizer='adam', loss='mae')

# train
print("Training.. on " + str(X_train.shape[0]) + " samples")
history = model.fit(X_train, y_train, epochs=500, batch_size=3200, validation_split=0.3, shuffle=False, verbose='1')

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training Loss (MSE)', 'Validation Loss (MSE)'], loc='upper left')
plt.savefig(db_path.rsplit('/', 1)[0]+"/loss.png")
plt.show()

print("Evaluate Model..")
model.evaluate(X_test, y_test, verbose=2)

print("y_train mean: " + str(y_train.mean()))
print("y_test mean: " + str(y_test.mean()))

print("Saving Model")
model.save(db_path.rsplit('/', 1)[0]+"/model")

# evaluation
# estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=20, verbose=0)
# kfold = KFold(n_splits=10)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))