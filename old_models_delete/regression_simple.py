# import os
# import time
# import matplotlib.pyplot as plt
# import numpy as np
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# import tensorflow.keras.backend as K
# from tensorflow.keras.regularizers import l2
# import sys
# import glob
# import pandas as pd
# from database import COLMAPDatabase
# from sklearn.model_selection import KFold
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
#
# # sample commnad to run on bath cloud servers, ogg .. etc
# # python3 regression_simple.py colmap_data/Coop_data/slice1/ML_data/ml_database.db 5 16384 1000 first_simple/
# # python3 regression_simple.py colmap_data/Coop_data/slice1/ML_data/ml_database.db 5 32768 500 second_simple/
#
# # this might cause problems when loading the model
# # def soft_acc(y_true, y_pred):
# #     return K.mean(K.equal(K.round(y_true), K.round(y_pred)))
#
# def split_data(features, target, test_percentage, randomize = False):
#     if(randomize):
#         print("Randomizing data")
#         union = np.c_[features, target]
#         np.random.shuffle(union)
#         features = union[:, 0:128]
#         target = union[:, 128]
#     rows_no = features.shape[0] #or test , same thing
#     train_percentage = 1 - test_percentage
#     train_max_idx = int(np.floor(rows_no * train_percentage))
#     X_train = features[0 :  train_max_idx , :]
#     y_train = target[0 : train_max_idx]
#     X_test = features[train_max_idx : , :]
#     y_test = target[train_max_idx :]
#     return X_train, y_train, X_test, y_test
#
# print("Running Script..!")
#
# db_path = sys.argv[1]
# num_folds = int(sys.argv[2])
# batch_size = int(sys.argv[3])
# epochs = int(sys.argv[4])
# model_base_name = sys.argv[5]
# base_path = "colmap_data/Coop_data/slice1/ML_data/results/"
# os.makedirs(base_path+model_base_name)
# base_path = base_path+model_base_name
#
# print("num_folds: " + str(num_folds))
# print("batch_size: " + str(batch_size))
# print("epochs: " + str(epochs))
#
# ml_db = COLMAPDatabase.connect_ML_db(db_path)
#
# sifts_scores = ml_db.execute("SELECT sift, score FROM data").fetchall() #guarantees same order
#
# all_sifts = (COLMAPDatabase.blob_to_array(row[0] , np.uint8) for row in sifts_scores)
# all_sifts = np.array(list(all_sifts))
#
# all_scores = (row[1] for row in sifts_scores)
# all_scores = np.array(list(all_scores))
#
# print("Splitting data into test/train..")
#
# # X_train, y_train, X_test, y_test = split_data(all_sifts, all_scores, 0.3, randomize = True)
# X_train, X_test, y_train, y_test = train_test_split(all_sifts, all_scores, test_size=0.2, shuffle=True, random_state=42)
#
# print("Total Training Size: " + str(X_train.shape[0]))
# print("Total Test Size: " + str(X_test.shape[0]))
#
# # standard scaling - mean normalization
# X_train = ( X_train - X_train.mean() ) / X_train.std()
# X_test = ( X_test - X_test.mean() ) / X_test.std()
# # min-max normalization
# y_train = ( y_train - y_train.min() ) / ( y_train.max() - y_train.min() )
# y_test = ( y_test - y_test.min() ) / ( y_test.max() - y_test.min() )
#
# print("Saving unseen test data..")
# np.save(base_path+"X_test", X_test)
# np.save(base_path+"y_test", y_test)
#
# print("y_train mean: " + str(y_train.mean()))
# print("y_test mean: " + str(y_test.mean()))
#
# eval_scores = []
# mse_scores_train = []
# mse_scores_test = []
# kfold = KFold(n_splits = num_folds, shuffle = True, random_state=42)
# fold_no = 1
# for train, test in kfold.split(X_train):
#     print('------------------------------------------------------------------------')
#     print(f'Training for fold {fold_no} ...')
#
#     # create model
#     print("Creating model")
#     model = Sequential()
#     # in keras the first layer is a hidden layer too, so input dims is OK here
#     # TODO: maybe change the activation function to 'sigmoid from relu' as your score will be always less than 1 and more than 0 (after normalization)?
#     model.add(Dense(128, input_dim=128, kernel_initializer='normal', activation='sigmoid')) #I know all input values will be positive at this point (SIFT)
#     model.add(Dense(1, kernel_initializer='normal', activation='sigmoid')) #TODO: relu might be more appropriate (?) here (since score can never be negative)
#
#     # Compile model
#     model.compile(optimizer='adam', loss='mse')
#
#     # train
#     print("Training.. on " + str(X_train[train].shape[0]) + " samples")
#     history = model.fit(X_train[train], y_train[train],
#                         validation_data=(X_train[test], y_train[test]),
#                         epochs=epochs, batch_size=batch_size,
#                         verbose=0)
#
#     print(history.history.keys())
#     # "Loss"
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model Loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Training Loss (MSE)', 'Validation Loss (MSE)'], loc='upper left')
#     plt.savefig(base_path + "loss_"+str(fold_no)+".png")
#
#     print("Evaluating..")
#     score = model.evaluate(X_train[test], y_train[test], verbose=0, batch_size=batch_size)
#     print(f"Fold score (MSE): {score}")
#     eval_scores.append(score)
#
#     pred_train = model.predict(X_train[train])
#     mse_train_val = metrics.mean_squared_error(y_train[train], pred_train)
#     print("MSE on Training Data (predict(train_data)): " + str(mse_train_val))
#     mse_scores_train.append(mse_train_val)
#
#     pred_test = model.predict(X_train[test])
#     mse_test_val = metrics.mean_squared_error(y_train[test], pred_test)
#     print("MSE on Testing Data (predict(test_data)): " + str(mse_test_val))
#     mse_scores_test.append(mse_test_val)
#
#     print("Saving model..")
#     model.save(base_path+"model_" + str(fold_no))
#
#     print("Saving model metrics..")
#     np.save(base_path + "fold_score_" + str(fold_no), score)
#     fold_no +=1
#
# mse_mean_eval = np.mean(eval_scores)
# mse_mean_train = np.mean(mse_scores_train)
# mse_mean_test = np.mean(mse_scores_test)
#
# print("MSE mean: " + str(mse_mean_eval))
# print("MSE mean (train): " + str(mse_mean_train))
# print("MSE mean (test): " + str(mse_mean_test))
#
# np.save(base_path+"mse_mean_eval", mse_mean_eval)
# np.save(base_path+"mse_mean_train", mse_mean_train)
# np.save(base_path+"mse_mean_test", mse_mean_test)
#
# print("Done!")
#
# # print("Evaluate Model..")
# # model.evaluate(X_test, y_test, verbose=2)
# #
# # print("y_train mean: " + str(y_train.mean()))
# # print("y_test mean: " + str(y_test.mean()))
# #
# # print("Saving Model")
# # model.save(db_path.rsplit('/', 1)[0]+"/model")
#
# # evaluation
# # estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=20, verbose=0)
# # kfold = KFold(n_splits=10)
# # results = cross_val_score(estimator, X, Y, cv=kfold)
# # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
