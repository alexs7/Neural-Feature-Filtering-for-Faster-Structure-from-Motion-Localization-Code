import sys
import os
import colmap

db_path = sys.argv[1]
new_model_dir = sys.argv[2]
new_model_dir_text = sys.argv[3]

colmap.model_converter(db_path, new_model_dir, new_model_dir_text)
