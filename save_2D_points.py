import cv2
import os

def save_debug_image_simple(image_gt_path, original_keypoints_xy, predicted_keypoints_xy, comparison_data_path, name):
    query_image_file = cv2.imread(image_gt_path)
    verif_img = query_image_file.copy()  # need a copy here
    for kp in original_keypoints_xy: # all (blue)
        kp = kp.astype(int)
        cv2.circle(verif_img, (kp[0], kp[1]), 4, (0, 0, 255), -1)  # red (all)
    for kp in predicted_keypoints_xy:  # only positive ones (green)
        kp = kp.astype(int)
        cv2.circle(verif_img, (kp[0], kp[1]), 4, (0, 255, 0), -1)  # green (the predicted ones)

    # Save image to disk
    verif_img_path = os.path.join(comparison_data_path, name)
    cv2.imwrite(verif_img_path, verif_img)