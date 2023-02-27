# This file will calculate the pose comaprison between the ones
# I calculated and the ones from COLMAP
# NOTE: run after ransac_comparison.py
import pdb

import numpy as np
from query_image import read_images_binary, get_query_image_global_pose_new_model, load_images_from_text_file, QuaternionFromMatrix

# 01/09/2022, This is used to return the error per image, using the new format!
def pose_evaluate_generic_comparison_model(query_pose, ground_truth_pose, scale):
    # camera center errors
    q_pose_cntr = -query_pose[0:3, 0:3].transpose().dot(query_pose[0:3, 3])
    gt_pose_cntr = -ground_truth_pose[0:3, 0:3].transpose().dot(ground_truth_pose[0:3, 3])
    # multiplying by scale will return the distance in (m) in the other dataset (ARCore or CMU or ...)
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
def pose_evaluate_generic_comparison_model_Maa(query_poses, gt_poses, degenerate_names, scale):
    # I use two different sets of thresholds over rotation and translation. Change this according to your needs
    assert len(query_poses) == len(gt_poses)
    thresholds_q = np.linspace(1, 10, 10)
    thresholds_t = np.geomspace(0.2, 5, 10) #np.geomspace(0.2, 5, 10), same as np.logspace(np.log10(0.2), np.log10(5), num=10, base=10.0)
    image_pose_errors = {}
    for image_name, _ in query_poses.items(): #can be refactored here
        if(len(degenerate_names) > 0 and image_name in degenerate_names):
            continue
        q_pose = query_poses[image_name][0]
        gt_pose = gt_poses[image_name]

        # camera center errors
        T = -q_pose[0:3, 0:3].transpose().dot(q_pose[0:3, 3])
        T_gt = -gt_pose[0:3, 0:3].transpose().dot(gt_pose[0:3, 3])

        q = QuaternionFromMatrix(q_pose[0:3, 0:3])
        q_gt = QuaternionFromMatrix(gt_pose[0:3, 0:3])

        # q, T are the estimated ones
        err_q, err_t = ComputeErrorForOneExample(q_gt, T_gt, q, T, scale)
        image_pose_errors[image_name] = [err_q, err_t] #this will not include the degenerate cases

    mAA, valid_images = ComputeMaa([v[0] for v in image_pose_errors.values()], [v[1] for v in image_pose_errors.values()], thresholds_q, thresholds_t, list(image_pose_errors.keys()))

    return mAA, valid_images

# (09/09/2022) from: https://www.kaggle.com/code/eduardtrulls/imc2022-training-data?scriptVersionId=92062607
def ComputeMaa(err_q, err_t, thresholds_q, thresholds_t, image_names):
    '''Compute the mean Average Accuracy at different tresholds, for one scene.'''

    assert len(err_q) == len(err_t)
    assert len(image_names) == len(err_t) #or err_q
    valid_imgs = np.array([])

    acc, acc_q, acc_t = [], [], []
    for th_q, th_t in zip(thresholds_q, thresholds_t):
        # In this loop the poses' errors are compared to the thresholds' values and added only if
        # both are True (are under that threshold value). Then it sums them and divides by
        # the number of total errors poses' errors to get a percentage (you need to multiply by 100 for the actual %)
        # acc the higher the better. acc_q is just for rotation, and acc_t for metric translation
        acc += [(np.bitwise_and(np.array(err_q) < th_q, np.array(err_t) < th_t)).sum() / len(err_q)]
        acc_q += [(np.array(err_q) < th_q).sum() / len(err_q)]
        acc_t += [(np.array(err_t) < th_t).sum() / len(err_t)]

        valid_img_idx = np.bitwise_and(np.array(err_q) < th_q, np.array(err_t) < th_t)
        # The valid_imgs are images that passed a certain threshold. To avoid duplicates
        # we use np.unique. duplicates will happen from the most accurate ones as
        # they automatically belong to all the thresholds
        valid_imgs = np.append(valid_imgs, [np.array(image_names)[np.where(valid_img_idx)]])

    # The main you can use for overall accuracy is np.mean(acc), maybe plot np.array(acc) and the rest?
    # np.unique(valid_imgs) is the images without duplicates from all threshold
    return [np.mean(acc), np.array(acc), np.array(acc_q), np.array(acc_t)] , np.unique(valid_imgs)

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