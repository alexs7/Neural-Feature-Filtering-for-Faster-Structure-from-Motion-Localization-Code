import os
from os import path
from pickle import dump
import shutil
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorboard_config import get_Tensorboard_dir
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.callbacks import TensorBoard
import sys
from custom_callback import getModelCheckpointCombined, getEarlyStoppingCombined
from data import getCombinedData

metrics = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='binary_accuracy'),
    # keras.metrics.Precision(name='precision'),
    # keras.metrics.Recall(name='recall'),
    # keras.metrics.AUC(name='auc'),
    # keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    keras.metrics.MeanSquaredError(name="mean_squared_error"),
    keras.metrics.MeanAbsoluteError(name="mean_absolute_error"),
    # keras.metrics.MeanAbsolutePercentageError(name="mean_absolute_percentage_error"),
    # keras.metrics.CosineSimilarity(name="cosine_similarity"),
    keras.metrics.RootMeanSquaredError(name="root_mean_squared_error")
]

# sample commnad to run on bath cloud servers, ogg .. etc
# python3 combined_models_4.py colmap_data/Coop_data/slice1/ 32768 1000 Extended_CMU_slice3 score_per_image ( or colmap_data/CMU_data/slice3/ )

base_path = sys.argv[1]
db_path = os.path.join(base_path, "ML_data/ml_database_all.db")
batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])
name = "combined_"+sys.argv[4]
score_to_train_on = sys.argv[5] #score_per_image, score_per_session, score_visibility
name = name + "_" + score_to_train_on

log_dir = get_Tensorboard_dir(name)
if(path.exists(log_dir)):
    print("Deleting: " + log_dir)
    shutil.rmtree(log_dir)
early_stop_model_save_dir = os.path.join(log_dir, "early_stop_model")
model_save_dir = os.path.join(log_dir, "model")

print("TensorBoard log_dir: " + log_dir)
tensorboard_cb = TensorBoard(log_dir=log_dir)
print("Early_stop_model_save_dir log_dir: " + early_stop_model_save_dir)
mc_callback = getModelCheckpointCombined(early_stop_model_save_dir)
es_callback = getEarlyStoppingCombined()
all_callbacks = [tensorboard_cb, mc_callback, es_callback]

print("Running Script..!")
print(name)

print("Batch_size: " + str(batch_size))
print("Epochs: " + str(epochs))

print("Loading data..")
sift_vecs, scores, classes = getCombinedData(db_path, score_name = score_to_train_on)

# These will overwrite the plots per dataset - but it is fine, it is the
# same plots - for classification/regression etc, i.e. it is dataset dependent not network dependent
print("Saving graphs of the distribution of the mean SIFT vectors - before standard scaler")
plt.hist(sift_vecs.mean(axis=1), bins=50, alpha=0.6, color='b')
plt.savefig(os.path.join("plots/dist_plots/", name+'_dist_before_Standard_Scaler.png'))

scaler = StandardScaler()
scaler_transformed = scaler.fit(sift_vecs)
sift_vecs = scaler_transformed.transform(sift_vecs)

plt.cla()
print("Saving graphs of the distribution of the mean SIFT vectors - after standard scaler")
plt.hist(sift_vecs.mean(axis=1), bins=50, alpha=0.6, color='r')
plt.savefig(os.path.join("plots/dist_plots/", name+'_dist_after_Standard_Scaler.png'))

# minmax scaler
print("Scaling output to 0 - 1 range") # any score is from 0 - N, so just scale it to 0 - 1 and use a sigmoid
min_max_scaler = MinMaxScaler()
scores = min_max_scaler.fit_transform(scores.reshape(-1, 1))

# Create model
print("Creating model")

# in keras the first layer is a hidden layer too, so input dims is OK here
inputs = Input(shape=(128,))
layer1 = Dense(256, activation='relu')(inputs)
layer2 = Dense(256, activation='relu')(layer1)
layer3 = Dense(256, activation='relu')(layer2)
layer4 = Dense(256, activation='relu')(layer3)
layer5 = Dense(256, activation='relu')(layer4)
layer6 = Dense(256, activation='relu')(layer5)
layer7 = Dense(256, activation='relu')(layer6)
layer8 = Dense(256, activation='relu')(layer7)
# multiple outputs
regression = Dense(1, activation='sigmoid', name="regression")(layer8) #sigmoid here can be used since the output is from zero to one (MinMax)
classifier = Dense(1, activation='sigmoid', name="classifier")(layer8)
model = Model(inputs=inputs, outputs=[regression, classifier])
# Compile model
opt = keras.optimizers.Adam(learning_rate=1e-4)
# The loss here will be, binary_crossentropy for the binary classsifier
model.compile(optimizer=opt, loss={"regression": keras.losses.MeanAbsoluteError(), "classifier": keras.losses.BinaryCrossentropy()}, metrics=metrics)
model.summary()
# keras.utils.plot_model(model, os.path.join(log_dir, "model_drawing.png"), show_shapes=True) #must run pip install pydot, https://graphviz.gitlab.io/download

# Before training you should use a baseline model

# Train (or fit() )
# Just for naming's sake
X_train = sift_vecs
y_train = classes
history = model.fit(X_train,
                    {"regression": scores, "classifier": classes}, #training data
                    validation_split=0.3,
                    epochs=epochs,
                    shuffle=True,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=all_callbacks)

# Save model here
print("Saving model...")
model.save(model_save_dir)
# This has to happen here because by now "log_dir" will have been created by Tensorboard
print("Saving Scalers..")
scaler_save_path = os.path.join(log_dir, "scaler.pkl")
dump(scaler_transformed, open(scaler_save_path, 'wb'))
min_max_scaler_save_path = os.path.join(log_dir, "min_max_scaler.pkl")
dump(min_max_scaler, open(min_max_scaler_save_path, 'wb'))

print("Done!")

