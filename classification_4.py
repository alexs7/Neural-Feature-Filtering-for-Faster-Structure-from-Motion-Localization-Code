import os
import time
from data import getClassificationData
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
import sys
# from sklearn.model_selection import KFold
from custom_callback import getModelCheckpointBinaryClassification, getEarlyStoppingBinaryClassification

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
# python3 classification_4.py colmap_data/Coop_data/slice1/ 32768 900 ManyManyNodesLayersEarlyStopping (or CMU slices path)

base_path = sys.argv[1]
db_path = os.path.join(base_path, "ML_data/ml_database_all.db")
batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])
name = sys.argv[4]

MODEL_NAME = "BinaryClassification-{}-{}".format( name, time.ctime())

model_results_dir = "ML_data/results/{}".format(MODEL_NAME)
log_dir = os.path.join(base_path, model_results_dir)
early_stop_model_save_dir = os.path.join(log_dir, "early_stop_model")
model_save_dir = os.path.join(log_dir, "model")

print("TensorBoard log_dir: " + log_dir)
tensorboard_cb = TensorBoard(log_dir=log_dir)
print("Early_stop_model_save_dir log_dir: " + early_stop_model_save_dir)
mc_callback = getModelCheckpointBinaryClassification(early_stop_model_save_dir)
es_callback = getEarlyStoppingBinaryClassification()
all_callbacks = [tensorboard_cb, mc_callback, es_callback]

print("Running Script..!")
print(MODEL_NAME)

print("Batch_size: " + str(batch_size))
print("Epochs: " + str(epochs))

print("Loading data..")
sift_vecs, classes = getClassificationData(db_path)

# Create model
print("Creating model")
model = Sequential()
# in keras the first layer is a hidden layer too, so input dims is OK here
model.add(Dense(128, input_dim=128, activation='relu')) #Note: 'relu' here will be the same as 'linear' (default as all SIFT values are positive)
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
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
                    validation_split=0.2,
                    epochs=epochs,
                    shuffle=True,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=all_callbacks)

# Save model here
print("Saving model..")
model.save(model_save_dir)

print("Done!")

