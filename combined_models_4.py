import os
import time

from tensorboard_config import get_Tensorboard_dir

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
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
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    keras.metrics.MeanSquaredError(name="mean_squared_error"), keras.metrics.MeanAbsoluteError(name="mean_absolute_error"),
    keras.metrics.MeanAbsolutePercentageError(name="mean_absolute_percentage_error"), keras.metrics.CosineSimilarity(name="cosine_similarity"),
    keras.metrics.RootMeanSquaredError(name="root_mean_squared_error")
]

# sample commnad to run on bath cloud servers, ogg .. etc
# python3 combined_models_4.py colmap_data/Coop_data/slice1/ 32768 1000 ManyManyNodesLayersEarlyStoppingCombinedModel ( or colmap_data/CMU_data/slice3/ )

base_path = sys.argv[1]
db_path = os.path.join(base_path, "ML_data/ml_database_all.db")
batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])
name = "combined_"+sys.argv[4]

log_dir = get_Tensorboard_dir(name)
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
sift_vecs, scores, classes = getCombinedData(db_path, score_name = "score_per_image")

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
classifier = Dense(1, activation='sigmoid', name="classifier")(layer8)
regression = Dense(1, activation='sigmoid', name="regression")(layer8) #sigmoid here can be used since the output is from zero to one (MinMax)
model = Model(inputs=inputs, outputs=[regression, classifier])
# Compile model
opt = keras.optimizers.Adam(learning_rate=3e-4)
# The loss here will be, binary_crossentropy
model.compile(optimizer=opt, loss={"regression": keras.losses.MeanAbsoluteError(), "classifier": keras.losses.BinaryCrossentropy()}, loss_weights=[0.5, 0.5], metrics=metrics)
model.summary()
# keras.utils.plot_model(model, os.path.join(log_dir, "model_drawing.png"), show_shapes=True) #must run pip install pydot, https://graphviz.gitlab.io/download

# Before training you should use a baseline model

# Train (or fit() )
# Just for naming's sake
X_train = sift_vecs
y_train = classes
history = model.fit(X_train,
                    {"regression": scores, "classifier": classes}, #training data
                    validation_split=0.2,
                    epochs=epochs,
                    shuffle=True,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=all_callbacks)

# Save model here
print("Saving model...")
model.save(model_save_dir) #double check with keras.models.load_model(model_save_path) in debug

print("Done!")

