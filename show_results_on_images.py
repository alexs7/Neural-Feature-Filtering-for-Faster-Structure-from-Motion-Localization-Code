# This script will load draw on the query images which points are mathcable and which ones are not.
# Will use a number of models including NN (the classifier only), PM, MnM
# This file will use the gt db as we need the data from the gt (query) images to display on the query images that data is not in live db
import os
import sys
import cv2
import numpy as np
from joblib import load
from tensorflow import keras
from pathlib import Path
from tqdm import tqdm
from database import COLMAPDatabase
from query_image import load_images_from_text_file, get_keypoints_xy, get_image_id, get_queryDescriptors, clear_folder, get_keypoints_data, \
    get_image_name_only_with_extension
from save_2D_points import save_debug_image_simple

base_path = sys.argv[1]
print("Base path: " + base_path)
mnm_base_path = sys.argv[2] # this is that data generated from format_data_for_match_no_match.py
gt_images_path = [x for x in Path(os.path.join(base_path,"gt/images/")).iterdir() if x.is_dir()][0] #i.e. colmap_data/CMU_data/slice3/gt/images/session_7
print("gt_images_path: " + str(gt_images_path))
nn_model_path = sys.argv[3] # i.e. "colmap_data/tensorboard_results/classification_Extended_CMU_slice3/early_stop_model/"
pm_no_samples = sys.argv[4] # the samples the model was trained on
mnm_no_samples = sys.argv[5] # the samples the model was trained on
debug_images_path = os.path.join(base_path, "ML_data", "debug_images")
sklearn_model_path = os.path.join(os.path.join(base_path, "predicting_matchability_comparison_data"), f"rforest_{pm_no_samples}.joblib")
mnm_model_path = os.path.join(base_path, "match_or_no_match_comparison_data", f"Trained model {mnm_no_samples}.xml")

clear_folder(debug_images_path)

# for PM and NN
db_gt_path = os.path.join(base_path, "gt/database.db")
db_gt = COLMAPDatabase.connect(db_gt_path)
# For MnM
gt_db_mnm_path = os.path.join(mnm_base_path, "gt/database.db")
gt_db_mnm = COLMAPDatabase.connect(gt_db_mnm_path)

query_images_path = os.path.join(base_path, "gt/query_name.txt")  # these are the same anw, as mnm
query_images_names = load_images_from_text_file(query_images_path)

# load PM, MnM, and NN model
# PM
rf = load(sklearn_model_path)
# MnM
cpp_original_model = cv2.ml.RTrees_load(mnm_model_path)
# NN
classification_model = keras.models.load_model(nn_model_path)

for name in tqdm(query_images_names):
    image_id = get_image_id(db_gt, name)
    original_image_path = os.path.join(gt_images_path, get_image_name_only_with_extension(name))

    # PM
    # Note the model trained on the 500 samples performs worse visually on the model that
    # was trained on the 2000 samples
    keypoints_xy_pm = get_keypoints_xy(db_gt, image_id)
    queryDescriptors = get_queryDescriptors(db_gt, image_id)
    predictions = rf.predict(queryDescriptors)
    output_filename = f"pm_{get_image_name_only_with_extension(name)}"
    save_debug_image_simple(original_image_path, keypoints_xy_pm, keypoints_xy_pm[predictions == 1], debug_images_path, output_filename)

    # MnM
    query_image_file = cv2.imread(original_image_path)
    keypoints_data = get_keypoints_data(gt_db_mnm, image_id, query_image_file) #note which db to use here, the OpenCV one
    keypoints_xy_mnm = keypoints_data[:, 0:2]
    _, predictions = cpp_original_model.predict(keypoints_data)
    predictions = np.array(predictions.ravel()).astype(np.uint8)
    output_filename = f"mnm_{get_image_name_only_with_extension(name)}"
    save_debug_image_simple(original_image_path, keypoints_xy_mnm, keypoints_xy_mnm[predictions == 1], debug_images_path, output_filename)

    # NN - all predicted
    keypoints_xy_nn = get_keypoints_xy(db_gt, image_id)
    queryDescriptors = get_queryDescriptors(db_gt, image_id)

    classifier_predictions = classification_model.predict_on_batch(queryDescriptors)
    matchable_desc_indices = np.where(classifier_predictions > 0.5)[0]
    keypoints_xy_nn_pred_all = keypoints_xy_nn[matchable_desc_indices]
    output_filename = f"nn_pred_all_{get_image_name_only_with_extension(name)}"
    save_debug_image_simple(original_image_path, keypoints_xy_nn, keypoints_xy_nn_pred_all, debug_images_path, output_filename)

    # NN - top 10% as in paper
    top_no = 10
    len_descs = queryDescriptors.shape[0]
    percentage_num = int(len_descs * top_no / 100)
    classification_sorted_indices = classifier_predictions[:, 0].argsort()[::-1]
    keypoints_xy_nn_pred_top = keypoints_xy_nn[classification_sorted_indices][0:percentage_num,:]
    output_filename = f"nn_pred_top_{get_image_name_only_with_extension(name)}"
    save_debug_image_simple(original_image_path, keypoints_xy_nn, keypoints_xy_nn_pred_top, debug_images_path, output_filename)

print("Done!")