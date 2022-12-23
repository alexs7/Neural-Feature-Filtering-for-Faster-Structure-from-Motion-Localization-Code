import os
import shutil

def remove_folder_safe(folder_path):
    print(f"Deleting {folder_path}")
    if (os.path.exists(folder_path)):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)
    pass