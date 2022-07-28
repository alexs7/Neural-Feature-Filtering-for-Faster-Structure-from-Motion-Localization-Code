import os
import sys
import cv2
import numpy as np

def save_projected_points_PM(image_gt_path, comaprison_data_path, all_kp, neg_kp):
    red = (0, 0, 255)
    green = (0, 255, 0)

    image = cv2.imread(image_gt_path)
    image_out_path = os.path.join(comaprison_data_path, query_image_no_folder)

    keypoints_xy_descs_all = np.loadtxt(sift_path_all)
    keypoints_xy_all = keypoints_xy_descs_all[:, 0:2]

    keypoints_xy_descs_classified = np.loadtxt(sift_path_classified)
    keypoints_xy_classified = keypoints_xy_descs_classified[:, 0:2]

    # all will red
    for i in range(int(len(keypoints_xy_descs_all))):
        x = int(keypoints_xy_all[i][0])
        y = int(keypoints_xy_all[i][1])
        center = (x, y)
        cv2.circle(image, center, 3, red, -1)

    # classified (godd ones) will be blue
    for i in range(int(len(keypoints_xy_descs_classified))):
        x = int(keypoints_xy_classified[i][0])
        y = int(keypoints_xy_classified[i][1])
        center = (x, y)
        cv2.circle(image, center, 3, blue, -1)

    cv2.imwrite(image_out_path, image)
