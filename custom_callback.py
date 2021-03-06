from keras.callbacks import ModelCheckpoint, EarlyStopping

def getModelCheckpointBinaryClassification(checkpoint_filepath):
    # https://stackoverflow.com/questions/61505749/tensorflowcan-save-best-model-only-with-val-acc-available-skipping
    return ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_binary_accuracy', mode='max', save_best_only=True, verbose=1)

def getEarlyStoppingBinaryClassification():
    # why use loss here: https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    return EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

def getModelCheckpointRegression(checkpoint_filepath):
    # https://stackoverflow.com/questions/61505749/tensorflowcan-save-best-model-only-with-val-acc-available-skipping
    # here you can also use "val_mean_absolute_error", "val_mean_squared_error", or "val_loss" - debatable
    return ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

def getEarlyStoppingRegression():
    # why use loss here: https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    return EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50) #can be the same as above if the loss function is the same

# for combined model
def getModelCheckpointCombined(checkpoint_filepath):
    # https://stackoverflow.com/questions/61505749/tensorflowcan-save-best-model-only-with-val-acc-available-skipping
    return ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

def getEarlyStoppingCombined():
    # why use loss here: https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    return EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50) #can be the same as above if the loss function is the same