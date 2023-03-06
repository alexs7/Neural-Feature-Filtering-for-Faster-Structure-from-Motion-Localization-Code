# This file will calculate the pose comparison between the ones I calculated and the ones from COLMAP
import numpy as np
from query_image import read_images_binary, get_query_image_global_pose_new_model, load_images_from_text_file, QuaternionFromMatrix

def pose_evaluate_generic_comparison_model(query_pose, ground_truth_pose, scale=1):
    # camera center errors
    q_pose_cntr = -query_pose[0:3, 0:3].transpose().dot(query_pose[0:3, 3])
    gt_pose_cntr = -ground_truth_pose[0:3, 0:3].transpose().dot(ground_truth_pose[0:3, 3])
    # multiplying by scale will return the distance in (m) in the other dataset (ARCore or CMU or ...)
    #  this is OK as both poses are in the same COLMAP coordinate system
    dist = scale * np.linalg.norm(q_pose_cntr - gt_pose_cntr)

    # rotations errors
    # from paper: Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions
    q_pose_R = query_pose[0:3, 0:3]
    gt_pose_R = ground_truth_pose[0:3, 0:3]
    # NOTE: arccos returns radians - but I convert it to angles
    a_rad = np.arccos((np.trace(np.dot(np.linalg.inv(gt_pose_R), q_pose_R)) - 1) / 2)
    a_deg = np.absolute(np.degrees(a_rad))

    return dist, a_deg

# (09/09/2022) from: https://www.kaggle.com/code/eduardtrulls/imc2022-training-data?scriptVersionId=92062607
def pose_evaluate_generic_comparison_model_Maa(est_poses, gt_poses, thresholds_q, thresholds_t, scale = 1):
    # I use two different sets of thresholds over rotation and translation. Change this according to your needs
    image_pose_errors = {}
    for image_name, est_values in est_poses.items():
        # different syntax from ExMaps as in ExMaps I also save other data along the pose
        est_pose = est_poses[image_name]
        gt_pose = gt_poses[image_name]

        # camera center errors
        T = -est_pose[0:3, 0:3].transpose().dot(est_pose[0:3, 3])
        T_gt = -gt_pose[0:3, 0:3].transpose().dot(gt_pose[0:3, 3])

        q = QuaternionFromMatrix(est_pose[0:3, 0:3])
        q_gt = QuaternionFromMatrix(gt_pose[0:3, 0:3])

        # q, T are the estimated ones
        err_q, err_t = ComputeErrorForOneExample(q_gt, T_gt, q, T, scale)
        image_pose_errors[image_name] = [err_q, err_t] #this will not include the degenerate cases

    # v[0] = err_q, v[1] = err_t
    mAA = ComputeMaa([v[0] for v in image_pose_errors.values()], [v[1] for v in image_pose_errors.values()], thresholds_q, thresholds_t)
    return mAA

# (09/09/2022) from: https://www.kaggle.com/code/eduardtrulls/imc2022-training-data?scriptVersionId=92062607
def ComputeMaa(err_q, err_t, thresholds_q, thresholds_t):
    '''Compute the mean Average Accuracy at different tresholds, for one scene.'''

    assert len(err_q) == len(err_t)

    acc, acc_q, acc_t = [], [], []
    for th_q, th_t in zip(thresholds_q, thresholds_t):
        # In this loop the poses' errors are compared to the thresholds' values and added only if
        # both are True (are under that threshold value). Then it sums them and divides by
        # the number of total errors poses' errors to get a percentage (you need to multiply by 100 for the actual %)
        # acc the higher the better. acc_q is just for rotation, and acc_t for metric translation
        acc += [(np.bitwise_and(np.array(err_q) < th_q, np.array(err_t) < th_t)).sum() / len(err_q)]
        acc_q += [(np.array(err_q) < th_q).sum() / len(err_q)]
        acc_t += [(np.array(err_t) < th_t).sum() / len(err_t)]

    return [np.mean(acc), np.array(acc), np.array(acc_q), np.array(acc_t)]

def ComputeErrorForOneExample(q_gt, T_gt, q, T, scale):
    '''Compute the error metric for a single example.

    The function returns two errors, over rotation and translation. These are combined at different thresholds by ComputeMaa in order to compute the mean Average Accuracy.'''
    eps = 1e-15

    q_gt_norm = q_gt / (np.linalg.norm(q_gt) + eps)
    q_norm = q / (np.linalg.norm(q) + eps)

    loss_q = np.maximum(eps, (1.0 - np.sum(q_norm * q_gt_norm) ** 2))
    err_q = np.arccos(1 - 2 * loss_q)

    # Apply the scaling factor for this scene.
    T_gt_scaled = T_gt * scale
    T_scaled = T * np.linalg.norm(T_gt) * scale / (np.linalg.norm(T) + eps)

    err_t = min(np.linalg.norm(T_gt_scaled - T_scaled), np.linalg.norm(T_gt_scaled + T_scaled))

    return err_q * 180 / np.pi, err_t