# This file will calculate the pose comaprison between the ones
# I calculated and the ones from COLMAP
# NOTE: run after ransac_comparison.py

import numpy as np
from query_image import read_images_binary, get_query_image_global_pose_new_model, load_images_from_text_file

def pose_evaluate(query_poses, gt_poses, scale= 1, verbose = False):
    trans_errors = []
    rotation_errors = []
    for image_name, _ in gt_poses.items():
        q_pose = query_poses[image_name]
        gt_pose = gt_poses[image_name]

        # camera center errors
        q_pose_cntr = q_pose[0:3, 0:3].transpose().dot(q_pose[0:3, 3])
        gt_pose_cntr = gt_pose[0:3, 0:3].transpose().dot(gt_pose[0:3, 3])
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
    if(verbose):
        print("Mean errors (trans, rots):")
        print("[ " + str(np.nanmean(trans_errors)) + " , " + str(np.nanmean(rotation_errors)) + " ]" )
    return trans_errors, rotation_errors