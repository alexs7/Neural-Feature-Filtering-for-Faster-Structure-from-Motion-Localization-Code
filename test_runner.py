import numpy as np
import pdb
import sys
import cv2
import os
import point3D_loader
import feature_extractor
from colmap_database import COLMAPDatabase

db_path = sys.argv[1]
points3D_txt_path = sys.argv[2]
query_image_folder_path = sys.argv[3]

db = COLMAPDatabase.connect(db_path)

print("Loading points 3D..")
points3D_objects = point3D_loader.parse_points3D(points3D_txt_path, db)

feature_extractor.run(db_path, query_image_folder_path)

query_images_features = feature_extractor.extract_features(query_image_folder_path, db)

des1 = query_images_features[0].descs
des2 = point3D_loader.get_descriptor_array(points3D_objects) # TODO: this is slow - move it to offline processing

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

des1.dtype = np.float32
des2.dtype = np.float32
matches = flann.knnMatch(des1, des2, k=2)

breakpoint()

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
