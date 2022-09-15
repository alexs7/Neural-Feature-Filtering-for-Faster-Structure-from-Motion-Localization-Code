# 15/09/2022 - major refactor
from ransac_comparison import run_comparison

def benchmark(benchmarks_iters, ransac_func, matches, query_images_names, K, val_idx=None):
    images_all_data = {}
    for i in range(benchmarks_iters):
        images_data = run_comparison(ransac_func, matches, query_images_names, K, val_idx=val_idx)
        # images_data = [est_pose, inliers_no, outliers_no, iterations, elapsed_time]
        images_all_data[i] = images_data  # [iteration] = images data from the iteration
    return images_all_data