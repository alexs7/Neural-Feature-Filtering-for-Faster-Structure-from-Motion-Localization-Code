import cv2

def save_debug_image_simple_ml(image_gt_path, original_keypoints_xy, predicted_keypoints_xy, output_path, pixel_size=4):
    query_image_file = cv2.imread(image_gt_path)
    verif_img = query_image_file.copy()  # need a copy here
    for kp in original_keypoints_xy:
        kp = kp.astype(int)
        cv2.circle(verif_img, (kp[0], kp[1]), pixel_size, (0, 0, 255), -1)  # red (all)
    for kp in predicted_keypoints_xy:  # only positive ones (green)
        kp = kp.astype(int)
        cv2.circle(verif_img, (kp[0], kp[1]), pixel_size-1, (0, 255, 0), -1)  # green (the predicted ones)
    # Save image to disk
    cv2.imwrite(output_path, verif_img)
    pass

def save_debug_image_simple_ml_red(image_gt_path, keypoints_xy, output_path, pixel_size=4):
    query_image_file = cv2.imread(image_gt_path)
    verif_img = query_image_file.copy()  # need a copy here
    for kp in keypoints_xy:
        kp = kp.astype(int)
        cv2.circle(verif_img, (kp[0], kp[1]), pixel_size, (0, 0, 255), -1)  # red (all)
    # Save image to disk
    cv2.imwrite(output_path, verif_img)
    pass

def save_debug_image_simple_ml_green(image_gt_path, keypoints_xy, output_path, pixel_size=4):
    query_image_file = cv2.imread(image_gt_path)
    verif_img = query_image_file.copy()  # need a copy here
    for kp in keypoints_xy:
        kp = kp.astype(int)
        cv2.circle(verif_img, (kp[0], kp[1]), pixel_size, (0, 255, 00), -1)  # red (all)
    # Save image to disk
    cv2.imwrite(output_path, verif_img)
    pass