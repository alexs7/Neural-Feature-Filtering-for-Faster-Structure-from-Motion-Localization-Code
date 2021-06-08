import os

def get_Tensorboard_dir(name):
    dir = os.path.join("colmap_data", "tensorboard_results")
    results_dir = os.path.join(dir, name)
    return results_dir