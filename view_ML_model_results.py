import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
from tensorflow import keras
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb

# How to run
# python3 view_ML_model_results.py colmap_data/Coop_data/slice1/ML_data/results/first/

base_path = sys.argv[1]

# Load and print errors
mse_mean_eval = np.load(base_path+"mse_mean_eval.npy")
rmse_mean_test = np.load(base_path+"rmse_mean_test.npy")
rmse_mean_train = np.load(base_path+"rmse_mean_train.npy")

print("MSE averaged on all folds: " + str(mse_mean_eval))
print("RMSE averaged on all folds (tests fold): " + str(rmse_mean_test))
print("RMSE averaged on all folds (train fold): " + str(rmse_mean_train))

model_path = base_path+"model/"
model = keras.models.load_model(model_path)

# Load unseen data during training
X_test = np.load(base_path+"X_test.npy")
y_test = np.load(base_path+"y_test.npy")

print("Predicting...")
pred_test = model.predict(X_test)
pred_test = ( pred_test - pred_test.min() ) / ( pred_test.max() - pred_test.min() )

# Just for clarity
ground_truth = y_test
pred_truth = pred_test

print("Plotting..")
plt.cla()
plt.axis([0, 1, 0, 14000])
plt.hist(ground_truth, bins=500)
plt.savefig(base_path + "hist_gt.png")
plt.hist(pred_truth, bins=500)
plt.savefig(base_path + "hist_pt.png")

for i in range(0, 100):
    idx = np.random.choice(np.arange(len(y_test)), 80, replace=False)
    plt.cla()
    plt.plot(ground_truth[idx], label='GT')
    plt.plot(pred_truth[idx], label='PT')
    plt.savefig(base_path + "plot_"+str(i)+".png")

print("Done")




