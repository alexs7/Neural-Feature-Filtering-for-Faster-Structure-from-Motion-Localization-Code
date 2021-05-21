from keras.callbacks import ModelCheckpoint, EarlyStopping

def getModelCheckpointBinaryClassification(checkpoint_filepath):
    # https://stackoverflow.com/questions/61505749/tensorflowcan-save-best-model-only-with-val-acc-available-skipping
    return ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_binary_accuracy', mode='max', save_best_only=True, verbose=1)

def getEarlyStoppingBinaryClassification():
    # why use loss here: https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    return EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
