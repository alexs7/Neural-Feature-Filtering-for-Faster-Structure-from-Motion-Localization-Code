# 15/09/2022 - major refactor
from ransac_comparison import run_comparison

def benchmark(ransac_func, matches, query_images_names, K, val_idx=None):
    images_data = run_comparison(ransac_func, matches, query_images_names, K, val_idx=val_idx)
    return images_data