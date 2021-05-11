import tensorflow as tf
from tensorflow import keras
from database import COLMAPDatabase
from parameters import Parameters
import numpy as np
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_query_images_pose_from_images, get_intrinsics_from_camera_bin
from point3D_loader import read_points3d_default, get_points3D_xyz
from feature_matching_generator_ML import feature_matcher_wrapper
from ransac_prosac import ransac, ransac_dist, prosac
from get_scale import calc_scale_COLMAP_ARCORE
from benchmark import benchmark
import sys

model_dir = sys.argv[1]
model = keras.models.load_model(model_dir)