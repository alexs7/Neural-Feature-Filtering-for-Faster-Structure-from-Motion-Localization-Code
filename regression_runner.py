from tensorflow import keras
import sys

model_path = sys.argv[1]
model = keras.models.load_model(model_path)

