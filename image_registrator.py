import sys
import os
import colmap
import time

db_path = sys.argv[1]
query_images_dir = sys.argv[2]
image_list_file = sys.argv[3]
existing_model_dir = sys.argv[4]
new_model_dir = sys.argv[5]

colmap.feature_extractor(db_path, query_images_dir, image_list_file)
colmap.vocab_tree_matcher(db_path, image_list_file)
colmap.image_registrator(db_path, existing_model_dir, new_model_dir)