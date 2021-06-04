import os
from tensorflow import keras
import matplotlib.pyplot as plt
from database import COLMAPDatabase
import numpy as np
from feature_matching_generator_ML import get_queryDescriptors
from query_image import get_image_id
import sys

# This file is used to plot ML data.
# The models here are the best performing for classification and regression as of 28 May
# example commnad: "python3 plot_dist_ml.py colmap_data/Coop_data/slice1/ML_data/results/BinaryClassification-ManyManyNodesLayersEarlyStopping-Fri\ May\ 21\ 07\:46\:55\ 2021/early_stop_model/  colmap_data/Coop_data/slice1/ML_data/results/Regression-ManyManyNodesLayersEarlyStopping-Thu\ May\ 27\ 15\:17\:26\ 2021/early_stop_model/"
# TODO: For this code in this file you have to use the container 'ar2056_bath2020ssh' in weatherwax, ssh root@172.17.0.13 (or whatever IP it is)
# This is because the method predict_on_batch() needs the GPUs for speed
class_model_dir = sys.argv[1]
regression_model_dir = sys.argv[2]

print("Loading Model..")
class_model = keras.models.load_model(class_model_dir)
regression_model = keras.models.load_model(regression_model_dir)

db_gt_path = "colmap_data/Coop_data/slice1/ML_data/original_data/gt/database.db"
db_gt = COLMAPDatabase.connect(db_gt_path)  # you need this database to get the query images descs as they do not exist in the live db!

# load data generated from "prepare_comaprison_data.py"
print("Loading Data..")
points3D_info = np.load('colmap_data/Coop_data/slice1/ML_data/avg_descs_xyz_ml.npy').astype(np.float32)
train_descriptors_live = points3D_info[:, 0:128]
query_images_ground_truth_poses = np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/query_images_ground_truth_poses.npy", allow_pickle=True).item()
localised_query_images_names = np.ndarray.tolist(np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/localised_query_images_names.npy"))
points3D_xyz_live = np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/points3D_xyz_live.npy")  # can also pick them up from points3D_info
K = np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/K.npy")
scale = np.load("colmap_data/Coop_data/slice1/ML_data/prepared_data/scale.npy")

# plotting starts here
print("Plotting..")

all_descs = np.empty([0, 128])
for i in range(len(localised_query_images_names)):
    query_image = localised_query_images_names[i]
    print("Getting data for image " + str(i + 1) + "/" + str(len(localised_query_images_names)) + ", " + query_image)
    image_id = get_image_id(db_gt, query_image)
    queryDescriptors = get_queryDescriptors(db_gt, image_id)
    all_descs = np.r_[all_descs, queryDescriptors]

regression_scores = regression_model.predict_on_batch(all_descs)

print("Saving graph..")
plt.cla()
plt.hist(regression_scores, bins=1000, label='Regression predicted values')
plt.legend(loc="upper left")
plt.xlabel("Predictions")
plt.ylabel("SIFT descs")
plt.savefig(os.path.join("/home/fullpipeline/plots/", "regression_predicted_values.png"))