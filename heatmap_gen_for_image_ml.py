# This script will generate 4 images per query images from session folders.
# original one, one with all the features, and the one with the selected features.
# this evaluates the NNs visually
# as of now it only works for the classifier all, it needs to be adjusted for the regressor model
# example command:
# python3 heatmap_gen_for_image_ml.py colmap_data/CMU_data/slice3/ colmap_data/tensorboard_results/classification_Extended_CMU_slice3/early_stop_model/ plots/cmu_slice3_comparison_gt_images session_7

import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import sys
import cv2
from os import path
import numpy as np
from tensorflow import keras
from database import COLMAPDatabase
from feature_matching_generator_ML import get_image_id, get_keypoints_xy, get_queryDescriptors
from query_image import read_images_binary, get_intrinsics_from_camera_bin, get_images_names_bin

base_path = sys.argv[1] # i.e colmap_data/CMU_data/slice3/
model_path = sys.argv[2] # i.e best performing model, colmap_data/tensorboard_results/classification_Extended_CMU_slice3/early_stop_model/
output_path_path = sys.argv[3] # i.e plots/slice3_comparison_gt_images (this has to be created manually)
gt_images_folder = sys.argv[4] #i.e session_7
gt_session_name = sys.argv[4] #i.e duplicate as above

if(path.exists(output_path_path)): #cleaning up old files and remaking dir
    print("Deleting: " + output_path_path)
    shutil.rmtree(output_path_path)
    os.mkdir(output_path_path)
else:
    os.mkdir(output_path_path)

model = keras.models.load_model(model_path)
images_path = os.path.join(base_path , "gt/model/images.bin")
points3D_path = os.path.join(base_path , "gt/model/points3D.bin")
cameras_path = os.path.join(base_path , "gt/model/cameras.bin")
db_query_path = os.path.join(base_path, "gt/database.db")
query_images_path = os.path.join(base_path, os.path.join("gt/images/", gt_images_folder))

images = read_images_binary(images_path)
all_images_names = get_images_names_bin(images_path) # these are localised images
K = get_intrinsics_from_camera_bin(cameras_path, 3) # 2 since we are talking about gt pics
db_gt = COLMAPDatabase.connect(db_query_path) #db_gt, db_query same thing
matchable_threshold = 0.5

counter = 0
print("Total images: " + str(len(all_images_names)))
for name in all_images_names:
    if(gt_session_name in name):
        print("Doing image: " + str(counter) + " " + name)
        image_id = get_image_id(db_gt, name)
        keypoints_xy = get_keypoints_xy(db_gt, image_id)
        print("Size of keypoints before: " + str(len(keypoints_xy)))
        queryDescriptors = get_queryDescriptors(db_gt, image_id)

        model_predictions = model.predict_on_batch(queryDescriptors)

        matchable_desc_indices = np.where(model_predictions > matchable_threshold)[0]  # matchable_desc_indices will index queryDescriptors/model_predictions
        matchable_desc_indices_length = matchable_desc_indices.shape[0]

        keypoints_xy_pred = keypoints_xy[matchable_desc_indices]
        print("Size of keypoints after prediction: " + str(len(keypoints_xy_pred)))

        filename = name.split("/")[1]
        original_image = os.path.join(query_images_path, filename)
        temp_name = filename.split(".")[0]
        filename_all_features = temp_name + "_all_features_"+str(len(keypoints_xy))+".jpg"
        filename_predicted_features = temp_name + "_predicted_features"+str(len(keypoints_xy_pred))+".jpg"
        filename_all_predicted_features = temp_name + "_all_predicted_features_"+str(len(keypoints_xy))+"_"+str(len(keypoints_xy_pred))+".jpg" #for showing both filtered and all features

        raw_output_image_path = os.path.join(output_path_path, filename)
        output_image_path_all_features = os.path.join(output_path_path, filename_all_features)
        output_image_path_predicted_features = os.path.join(output_path_path, filename_predicted_features)
        output_image_path_all_predicted_features = os.path.join(output_path_path, filename_all_predicted_features)

        raw_image = cv2.imread(original_image)
        image_all_features = cv2.imread(original_image)
        image_predicted_features = cv2.imread(original_image)
        image_all_predicted_features = cv2.imread(original_image)

        for i in range(len(keypoints_xy)):
            x = int(keypoints_xy[i][0])
            y = int(keypoints_xy[i][1])
            center = (x, y)
            cv2.circle(image_all_features, center, 6, (0, 0, 255), -1)

        for i in range(len(keypoints_xy_pred)):
            x = int(keypoints_xy_pred[i][0])
            y = int(keypoints_xy_pred[i][1])
            center = (x, y)
            cv2.circle(image_predicted_features, center, 6, (0, 255, 0), -1)

        # for both in the same image (showing filtered and all features on same image)
        for i in range(len(keypoints_xy)):
            x = int(keypoints_xy[i][0])
            y = int(keypoints_xy[i][1])
            center = (x, y)
            cv2.circle(image_all_predicted_features, center, 6, (0, 0, 255), -1)

        for i in range(len(keypoints_xy_pred)):
            x = int(keypoints_xy_pred[i][0])
            y = int(keypoints_xy_pred[i][1])
            center = (x, y)
            cv2.circle(image_all_predicted_features, center, 6, (0, 255, 0), -1)

        cv2.imwrite(raw_output_image_path, raw_image)
        cv2.imwrite(output_image_path_all_features, image_all_features)
        cv2.imwrite(output_image_path_predicted_features, image_predicted_features)
        cv2.imwrite(output_image_path_all_predicted_features, image_all_predicted_features)

        counter = counter + 1 #counter will return less images because only localised are returned. the folder contains also un-localised ones