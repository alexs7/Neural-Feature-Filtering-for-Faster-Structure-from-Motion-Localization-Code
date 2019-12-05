import sys
import os
import colmap
import time

db_path = sys.argv[1]
new_model_dir = sys.argv[2]
new_model_dir_text = sys.argv[3]

start = time.time()
colmap.model_converter(db_path, new_model_dir, new_model_dir_text)
end = time.time()
print("model_converter took (s): " + str(end-start))
