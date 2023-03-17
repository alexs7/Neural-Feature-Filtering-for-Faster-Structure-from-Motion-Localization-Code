import datetime
import os
import numpy as np
from custom_loss_nn import tweaked_loss
from database import COLMAPDatabase
from parameters import Parameters
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
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

def format_rows(sift, xy, blue, green, red, octave, angle, size, response, domOrientations, matched):
    sift_tf = tf.io.decode_raw(sift, tf.uint8)  # SIFT
    sift_tf = tf.reshape(sift_tf, [1,128])
    xy_tf = tf.io.decode_raw(xy, np.float64) #xy
    xy_tf = tf.reshape(xy_tf, [1,2])

    # Cast all tensors to float32, and reshape to 1x1 as needed
    sift_tf = tf.cast(sift_tf, tf.float32)
    xy_tf = tf.cast(xy_tf, tf.float32)

    blue = tf.reshape(blue, [1,1])
    green = tf.reshape(green, [1,1])
    red = tf.reshape(red, [1,1])
    octave = tf.reshape(octave, [1,1])
    angle = tf.reshape(angle, [1,1])
    size = tf.reshape(size, [1,1])
    response = tf.reshape(response, [1,1])
    domOrientations = tf.reshape(domOrientations, [1,1])

    blue = tf.cast(blue, tf.float32)
    green = tf.cast(green, tf.float32)
    red = tf.cast(red, tf.float32)
    octave = tf.cast(octave, tf.float32)
    angle = tf.cast(angle, tf.float32)
    size = tf.cast(size, tf.float32)
    response = tf.cast(response, tf.float32)
    domOrientations = tf.cast(domOrientations, tf.float32)

    features = tf.concat([sift_tf, xy_tf, blue, green, red, octave, angle, size, response, domOrientations], axis=1)
    features = tf.reshape(features, [138, 1])

    label = tf.reshape(matched, [1, 1])
    label = tf.cast(label, tf.float32)

    return features, label

def run(params):

    base_path = params.base_path
    batch_size = 4096
    epochs = 1000
    print("Batch_size: " + str(batch_size))
    print("Epochs: " + str(epochs))

    ml_path = os.path.join(base_path, "ML_data") #folder should exist from create_nf_training_data.py
    os.makedirs(ml_path, exist_ok=True)
    nn_save_path = os.path.join(ml_path, "classification_model") #to store the binary model
    os.makedirs(nn_save_path, exist_ok=True)
    log_folder_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_folder = os.path.join(ml_path, "tensorboard_logs", log_folder_name)
    print(f"Tensorboard log folder: {log_folder}")
    callbacks = [TensorBoard(log_dir=log_folder, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', profile_batch=2, embeddings_freq=1)]

    total_rows = 11940462
    train_dataset = tf.data.experimental.SqlDataset("sqlite", params.ml_database_all_opencv_sift_path,
                                              f"SELECT sift, xy, blue, green, red, octave, angle, size, response, domOrientations, matched FROM data WHERE (base == 1 OR live == 1) ORDER BY RANDOM() LIMIT {int(total_rows*0.7)}",
                                              (tf.string, tf.string, tf.int32, tf.int32, tf.int32, tf.int32, tf.float64, tf.float64, tf.float64, tf.int32, tf.int32))
    val_dataset = tf.data.experimental.SqlDataset("sqlite", params.ml_database_all_opencv_sift_path,
                                              f"SELECT sift, xy, blue, green, red, octave, angle, size, response, domOrientations, matched FROM data  WHERE (base == 1 OR live == 1) ORDER BY RANDOM() LIMIT -1 OFFSET {int(total_rows*0.7)}",
                                              (tf.string, tf.string, tf.int32, tf.int32, tf.int32, tf.int32, tf.float64, tf.float64, tf.float64, tf.int32, tf.int32))
    # mapp function to help with formatting
    train_dataset = train_dataset.map(format_rows)
    val_dataset = val_dataset.map(format_rows)
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    # print("Splitting dataset")
    # train_ds, val_ds = tf.keras.utils.split_dataset(formatted_dataset, left_size=0.7)

    # Create model
    print("Creating model")

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
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
        model.compile(optimizer=opt, loss=tweaked_loss, metrics=metrics)

    model.summary()

    # NOTE: Before training you should use a baseline model
    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, shuffle=True, verbose=1, callbacks=callbacks)

    # Save model here
    print("Saving model..")
    model.save(nn_save_path)
    print("Done!")

root_path = "/media/iNicosiaData/engd_data/"
dataset = sys.argv[1]

# To debug
# debug = True
# if(debug == True):
# np.set_printoptions(threshold=sys.maxsize, precision=5, suppress=True)
# tf.config.experimental_run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

if(dataset == "HGE" or dataset == "CAB" or dataset == "LIN"):
    base_path = os.path.join(root_path, "lamar", f"{dataset}_colmap_model")
    print("Base path: " + base_path)
    print("Loading data..")
    run(Parameters(base_path))

if(dataset == "CMU"):
    if(len(sys.argv) > 4):
        slices_names = [sys.argv[4]]
    else: # do all slices
        slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                        "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    for slice_name in slices_names:
        # overwrite paths
        base_path = f"/media/iNicosiaData/engd_data/cmu/{slice_name}/exmaps_data/"
        print("Base path: " + base_path)
        print("Loading data..")
        run(Parameters(base_path))

if(dataset == "RetailShop"):
    base_path = os.path.join(root_path, "retail_shop", "slice1")
    print("Base path: " + base_path)
    print("Loading data..")
    run(Parameters(base_path))
