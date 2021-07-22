# This script will generate 2 images per live images from session folders.
# The outpur folder needs to be manually created.
# It will fail at the base images, because they are not under the "live/images" folder - so you will get an error

import os
import sys
import cv2
import numpy as np
from tensorflow import keras
from database import COLMAPDatabase
from feature_matching_generator_ML import get_image_id, get_keypoints_xy, get_queryDescriptors
from query_image import read_images_binary, get_intrinsics_from_camera_bin, get_images_names_bin

base_path = sys.argv[1] # i.e colmap_data/CMU_data/slice3/
model_path = sys.argv[2] # i.e colmap_data/tensorboard_results/classification_Extended_CMU_slice3/early_stop_model/
output_path_path = sys.argv[3] # i.e plots/slice3_comparison_images (this has to be created manually)

model = keras.models.load_model(model_path)
images_path = os.path.join(base_path , "live/model/images.bin")
points3D_path = os.path.join(base_path , "live/model/points3D.bin")
cameras_path = os.path.join(base_path , "live/model/cameras.bin")
db_live_path = os.path.join(base_path , "live/database.db")
live_images_path = os.path.join(base_path , "live/images/")

images = read_images_binary(images_path)
all_images_names = get_images_names_bin(images_path) # these are localised images
K = get_intrinsics_from_camera_bin(cameras_path, 2) # 2 since we are talking about live pics
db_live = COLMAPDatabase.connect(db_live_path)
matchable_threshold = 0.5

counter = 0
print("Total images: " + str(len(all_images_names)))
for name in all_images_names:
    print("Doing image: " + str(counter) + " " + name)
    image_id = get_image_id(db_live, name)
    keypoints_xy = get_keypoints_xy(db_live, image_id)
    queryDescriptors = get_queryDescriptors(db_live, image_id)

    model_predictions = model.predict_on_batch(queryDescriptors)

    matchable_desc_indices = np.where(model_predictions > matchable_threshold)[0]  # matchable_desc_indices will index queryDescriptors/model_predictions
    matchable_desc_indices_length = matchable_desc_indices.shape[0]

    keypoints_xy_pred = keypoints_xy[matchable_desc_indices]

    original_image = os.path.join(live_images_path, name)

    # This is used for saving only
    if(len(name.split("/")) == 1):
        filename = name
        temp_name = filename.split(".")[0]
        filename_original = temp_name + "_original.jpg"
    if (len(name.split("/")) == 2):
        filename = name.split("/")[1]
        temp_name = filename.split(".")[0]
        filename_original = temp_name + "_original.jpg"

    output_image_path = os.path.join(output_path_path, filename)
    output_original_image_path = os.path.join(output_path_path, filename_original)

    image = cv2.imread(original_image)
    orginal_image = cv2.imread(original_image)

    for i in range(len(keypoints_xy_pred)):
        x = int(keypoints_xy_pred[i][0])
        y = int(keypoints_xy_pred[i][1])
        center = (x, y)
        cv2.circle(image, center, 6, (0, 0, 255), -1)

    for i in range(len(keypoints_xy)):
        x = int(keypoints_xy[i][0])
        y = int(keypoints_xy[i][1])
        center = (x, y)
        cv2.circle(orginal_image, center, 6, (0, 0, 255), -1)

    cv2.imwrite(output_image_path, image)
    cv2.imwrite(output_original_image_path, orginal_image)

    counter=counter+1