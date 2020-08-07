import numpy as np
from pose_evaluator import pose_evaluate
from ransac_comparison import run_comparison

def benchmark(benchmarks_iters, ransac, matches_base, query_images_names, K, query_images_ground_truth_poses, scale, val_idx=None):
    trans_errors_overall = []
    rot_errors_overall = []
    inlers_no = []
    outliers = []
    iterations = []
    time = []

    for i in range(benchmarks_iters):
        poses , data = run_comparison(ransac, matches_base, query_images_names, K, val_idx=val_idx)
        trans_errors, rot_errors = pose_evaluate(poses, query_images_ground_truth_poses, scale)

        inlers_no.append(data.mean(axis=0)[0])
        outliers.append(data.mean(axis=0)[1])
        iterations.append(data.mean(axis=0)[2])
        time.append(data.mean(axis=0)[3])
        trans_errors_overall.append(np.nanmean(trans_errors))
        rot_errors_overall.append(np.nanmean(rot_errors))

    inlers_no = np.array(inlers_no).mean()
    outliers = np.array(outliers).mean()
    iterations = np.array(iterations).mean()
    time = np.array(time).mean()
    trans_errors_overall = np.array(trans_errors_overall).mean()
    rot_errors_overall = np.array(rot_errors_overall).mean()

    return inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall

