import datetime
import os
import numpy as np
from SQLiteDataGenerator import DataGenerator
from custom_loss_nn import tweaked_loss
from data import getClassificationDataOpenCV
from parameters import Parameters
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
import sys

if tf.config.experimental.list_logical_devices('GPU'):
    print('GPU found')
    print("List GPUs:")
    print(tf.config.list_physical_devices('GPU'))
else:
    print("No GPU found")

def run(base_path, db_path):

    params = Parameters(base_path)

    ml_path = os.path.join(params.base_path, "ML_data") #folder should exist from create_nf_training_data.py
    os.makedirs(ml_path, exist_ok=True)
    nn_save_path = os.path.join(ml_path, "classification_model_bce") #to store the binary model
    os.makedirs(nn_save_path, exist_ok=True)
    log_folder_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_folder = os.path.join(ml_path, "tensorboard_logs", log_folder_name)
    callbacks = [TensorBoard(log_dir=log_folder, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', profile_batch=2, embeddings_freq=1)]
    print(f"Tensorboard log folder: {log_folder}")
    batch_size = 4096
    epochs = 1000
    print("Batch_size: " + str(batch_size))
    print("Epochs: " + str(epochs))

    if(db_path == "0"): #to use absolute path on server or my server locally
        data = getClassificationDataOpenCV(params.ml_database_all_opencv_sift_path) #locally
    else:
        data = getClassificationDataOpenCV(db_path) #server

    # split to val and train data
    val_size = int(data.shape[0] * 30 / 100)
    X_val = data[0:val_size, 0:138]
    y_val = data[0:val_size, 138].astype(np.float32)
    X_train = data[val_size:, 0:138]
    y_train = data[val_size:, 138].astype(np.float32)

    # converts batches so np.float32
    training_gen_data = DataGenerator(X_train, y_train, batch_size)
    validation_gen_data = DataGenerator(X_val, y_val, batch_size)

    # Create model
    print("Creating model")
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

    model = Sequential()
    # in keras the first layer is a hidden layer too, so input dims is OK here
    model.add(Dense(138, input_dim=138, activation='relu')) #Note: 'relu' here will be the same as 'linear' (default as all values are positive)
    model.add(Dense(276, activation='relu'))
    model.add(Dense(276, activation='relu'))
    model.add(Dense(276, activation='relu'))
    model.add(Dense(138, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=metrics)
    model.summary()

    # NOTE: Before training you should use a baseline model
    model.fit(training_gen_data, validation_data=validation_gen_data, epochs=epochs,
              shuffle=True, verbose=1, callbacks=callbacks)

    # Save model here
    print("Saving model..")
    model.save(nn_save_path)
    print("Done!")

root_path = "/media/iNicosiaData/engd_data/"
dataset = sys.argv[1]
db_path = sys.argv[2]

# To debug
# debug = True
# if(debug == True):
#     # np.set_printoptions(threshold=sys.maxsize, precision=5, suppress=True)
#     tf.config.experimental_run_functions_eagerly(True)
#     tf.data.experimental.enable_debug_mode()

if(dataset == "HGE" or dataset == "CAB" or dataset == "LIN"):
    base_path = os.path.join(root_path, "lamar", f"{dataset}_colmap_model")
    print("Base path: " + base_path)
    run(base_path, db_path)

if(dataset == "CMU"):
    if(len(sys.argv) > 3):
        slices_names = [sys.argv[3]]
    else: # do all slices
        slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                        "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    for slice_name in slices_names:
        # overwrite paths
        base_path = f"/media/iNicosiaData/engd_data/cmu/{slice_name}/exmaps_data/"
        print("Base path: " + base_path)
        run(base_path, db_path)

if(dataset == "RetailShop"):
    base_path = os.path.join(root_path, "retail_shop", "slice1")
    print("Base path: " + base_path)
    run(base_path, db_path)