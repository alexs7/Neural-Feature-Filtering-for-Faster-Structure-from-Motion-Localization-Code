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
# points3D_objects = point3D_loader.parse_points3D(points3D_txt_path, db)

query_features = feature_extractor.run(db_path, query_image_folder_path)
