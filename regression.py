import pdb
from sklearn import svm
import numpy as np

idxs = [156, 144, 192, 204, 168, 1, 4, 5, 90, 3, 111]
all_data = np.empty([0,129])

for idx in idxs:
    data = np.load("/home/alex/fullpipeline/colmap_data/Coop_data/slice1/live_model_training_data/training_data_"+str(idx)+".npy")
    all_data = np.r_[all_data, data]

X = all_data[:,0:128]
y = all_data[:,128]

regr = svm.SVR(kernel='linear')
regr.fit(X, y)

pdb.set_trace()