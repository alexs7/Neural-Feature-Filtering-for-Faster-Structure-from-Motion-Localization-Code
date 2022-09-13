# This file will calculate the pose comaprison between the ones
# I calculated and the ones from COLMAP
# NOTE: run after ransac_comparison.py

import numpy as np
from query_image import read_images_binary, get_query_image_global_pose_new_model, load_images_from_text_file, QuaternionFromMatrix

def pose_evaluate(query_poses, gt_poses, scale= 1):
    trans_errors = []
    rotation_errors = []
    for image_name, _ in query_poses.items():
        q_pose = query_poses[image_name]
        gt_pose = gt_poses[image_name]

        # camera center errors
        q_pose_cntr = -q_pose[0:3, 0:3].transpose().dot(q_pose[0:3, 3])
        gt_pose_cntr = -gt_pose[0:3, 0:3].transpose().dot(gt_pose[0:3, 3])
        # multiplying by scale will return the distance in (m) in the other dataset (ARCore or CMU or ...)
        dist = scale * np.linalg.norm(q_pose_cntr - gt_pose_cntr)
        trans_errors.append(dist)

        # rotations errors
        # from paper: Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions
        q_pose_R = q_pose[0:3, 0:3]
        gt_pose_R = gt_pose[0:3, 0:3]
        # NOTE: arccos returns radians - but I convert it to angles
        a_rad = np.arccos((np.trace(np.dot(np.linalg.inv(gt_pose_R), q_pose_R)) - 1) / 2)
        a_deg = np.degrees(a_rad)
        rotation_errors.append(a_deg)

    # Note: These might contain nan values!
    trans_errors = np.array(trans_errors)
    rotation_errors = np.array(rotation_errors)

    return trans_errors, rotation_errors

# This will return "image_pose_errors" ONLY! for the bucket results from WACV2022 feedback
def pose_evaluate_ml(query_poses, gt_poses, scale= 1):
    trans_errors = []
    rotation_errors = []
    image_pose_errors = {}
    for image_name, _ in query_poses.items():
        q_pose = query_poses[image_name]
        gt_pose = gt_poses[image_name]

        # camera center errors
        q_pose_cntr = -q_pose[0:3, 0:3].transpose().dot(q_pose[0:3, 3])
        gt_pose_cntr = -gt_pose[0:3, 0:3].transpose().dot(gt_pose[0:3, 3])
        # multiplying by scale will return the distance in (m) in the other dataset (ARCore or CMU or ...)
        dist = scale * np.linalg.norm(q_pose_cntr - gt_pose_cntr)
        trans_errors.append(dist)

        # rotations errors
        # from paper: Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions
        q_pose_R = q_pose[0:3, 0:3]
        gt_pose_R = gt_pose[0:3, 0:3]
        # NOTE: arccos returns radians - but I convert it to angles
        a_rad = np.arccos((np.trace(np.dot(np.linalg.inv(gt_pose_R), q_pose_R)) - 1) / 2)
        a_deg = np.degrees(a_rad)
        rotation_errors.append(a_deg)

        image_pose_errors[image_name] = np.array([dist, a_deg])

    return image_pose_errors

# 01/09/2022, This is used to return the error per image, using the new format!
def pose_evaluate_generic_comparison_model(query_poses, gt_poses, scale = 1):
    image_pose_errors = {}
    for image_name, _ in query_poses.items():
        q_pose = query_poses[image_name][0]
        gt_pose = gt_poses[image_name]

        # camera center errors
        q_pose_cntr = -q_pose[0:3, 0:3].transpose().dot(q_pose[0:3, 3])
        gt_pose_cntr = -gt_pose[0:3, 0:3].transpose().dot(gt_pose[0:3, 3])
        # multiplying by scale will return the distance in (m) in the other dataset (ARCore or CMU or ...)
        dist = scale * np.linalg.norm(q_pose_cntr - gt_pose_cntr)

        # rotations errors
        # from paper: Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions
        q_pose_R = q_pose[0:3, 0:3]
        gt_pose_R = gt_pose[0:3, 0:3]
        # NOTE: arccos returns radians - but I convert it to angles
        a_rad = np.arccos((np.trace(np.dot(np.linalg.inv(gt_pose_R), q_pose_R)) - 1) / 2)
        a_deg = np.degrees(a_rad)

        image_pose_errors[image_name] = np.array([dist, a_deg])
    return image_pose_errors

# (09/09/2022) from: https://www.kaggle.com/code/eduardtrulls/imc2022-training-data?scriptVersionId=92062607
def pose_evaluate_generic_comparison_model_Maa(query_poses, gt_poses, scale = 1):
    # I use two different sets of thresholds over rotation and translation. Change this according to your needs
    thresholds_q = np.linspace(1, 10, 10)
    thresholds_t = np.geomspace(0.2, 5, 10) #same as np.logspace(np.log10(0.2), np.log10(5), num=10, base=10.0)
    image_pose_errors = {}
    for image_name, _ in query_poses.items():
        q_pose = query_poses[image_name][0]
        gt_pose = gt_poses[image_name]

        # camera center errors
        T = -q_pose[0:3, 0:3].transpose().dot(q_pose[0:3, 3])
        T_gt = -gt_pose[0:3, 0:3].transpose().dot(gt_pose[0:3, 3])

        q = QuaternionFromMatrix(q_pose[0:3, 0:3])
        q_gt = QuaternionFromMatrix(gt_pose[0:3, 0:3])

        err_q, err_t = ComputeErrorForOneExample(q_gt, T_gt, q, T, scale)
        image_pose_errors[image_name] = [err_q, err_t]

    mAA = ComputeMaa([v[0] for v in image_pose_errors.values()], [v[1] for v in image_pose_errors.values()], thresholds_q, thresholds_t)
    print(f'    Mean average Accuracy: {mAA[0]:.05f}')
    return mAA

# (09/09/2022) from: https://www.kaggle.com/code/eduardtrulls/imc2022-training-data?scriptVersionId=92062607
def ComputeMaa(err_q, err_t, thresholds_q, thresholds_t):
    '''Compute the mean Average Accuracy at different tresholds, for one scene.'''

    assert len(err_q) == len(err_t)

    acc, acc_q, acc_t = [], [], []
    for th_q, th_t in zip(thresholds_q, thresholds_t):
        # In this loop the poses' errors are compared to the thresholds' values and added only if
        # both are True (are under that threshold value). Then it sums them and divides by
        # the number of total errors poses' errors to get a percentage
        # acc the higher the better. acc_q is just for rotation, and acc_t for metric translation
        acc += [(np.bitwise_and(np.array(err_q) < th_q, np.array(err_t) < th_t)).sum() / len(err_q)]
        acc_q += [(np.array(err_q) < th_q).sum() / len(err_q)]
        acc_t += [(np.array(err_t) < th_t).sum() / len(err_t)]

    # The main you can use for overall accuracy is np.mean(acc), maybe plot np.array(acc) and the rest?
    return np.mean(acc), np.array(acc), np.array(acc_q), np.array(acc_t)

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