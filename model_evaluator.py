import tensorflow as tf
from tensorflow import keras
from database import COLMAPDatabase
from parameters import Parameters
import numpy as np
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_query_images_pose_from_images, get_intrinsics_from_camera_bin
from point3D_loader import read_points3d_default, get_points3D_xyz
from feature_matching_generator_ML import feature_matcher_wrapper, feature_matcher_wrapper_ml
from ransac_prosac import ransac, ransac_dist, prosac
from get_scale import calc_scale_COLMAP_ARCORE
from benchmark import benchmark
import sys

# example commnad: "python3 model_evaluator.py colmap_data/Coop_data/slice1/ML_data/results/BinaryClassificationSimple-1620823885/model/"
class_model_dir = sys.argv[1]

class_model = keras.models.load_model(class_model_dir)

db_gt_path = "colmap_data/Coop_data/slice1/ML_data/after_epoch_data/after_epoch_database.db"
db_gt = COLMAPDatabase.connect(db_gt_path)  # you need this database to get the query images descs as they do not exist in the live db!

# load data generated from "prepare_comaprison_data.py"
print("Loading Data..")
train_descriptors_live = np.load('colmap_data/Coop_data/slice1/ML_data/after_epoch_data/original_live_data/avg_descs.npy').astype(np.float32)
query_images_ground_truth_poses = np.load("colmap_data/Coop_data/slice1/ML_data/comparison_data/query_images_ground_truth_poses.npy", allow_pickle=True).item()
localised_query_images_names = np.ndarray.tolist(np.load("colmap_data/Coop_data/slice1/ML_data/comparison_data/localised_query_images_names.npy"))
points3D_xyz_live = np.load("colmap_data/Coop_data/slice1/ML_data/comparison_data/points3D_xyz_live.npy")
K = np.load("colmap_data/Coop_data/slice1/ML_data/comparison_data/K.npy")
scale = np.load("colmap_data/Coop_data/slice1/ML_data/comparison_data/scale.npy")
random_matches = np.load("colmap_data/Coop_data/slice1/ML_data/comparison_data/random_matches.npy", allow_pickle=True).item()
vanillia_matches = np.load("colmap_data/Coop_data/slice1/ML_data/comparison_data/vanillia_matches.npy", allow_pickle=True).item()
# results data, check "prepare_comaprison_data.py" for order
random_matches_data = np.load("colmap_data/Coop_data/slice1/ML_data/comparison_data/random_matches_data.npy")
vanillia_matches_data = np.load("colmap_data/Coop_data/slice1/ML_data/comparison_data/vanillia_matches_data.npy")

# evaluation starts here
print("Feature matching using model..")
# db_gt, again because we need the descs from the query images
ratio_test_val = 0.9  # as previous publication
# top 80 ones - why 80 ?
top = 80  # top or random - here it is top, because I am using the models
model_matches = feature_matcher_wrapper_ml(class_model, db_gt, localised_query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, verbose=True, random_limit=top)