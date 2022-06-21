import sys
import cv2
import numpy as np

# Example comand for image "CMU_data/slice3/gt/images/session_7/img_01763_c0_1288792470557721us.jpg"
# python3 show_2D_points_predicting_matchability.py colmap_data/CMU_data/slice3/gt/images/session_7/img_01763_c0_1288792470557721us.jpg colmap_data/CMU_data/slice3/gt/images/session_7/img_01763_c0_1288792470557721us.sift_not_classified colmap_data/CMU_data/slice3/gt/images/session_7/img_01763_c0_1288792470557721us.sift_classified colmap_data/CMU_data/slice3/gt/images/session_7/img_01763_c0_1288792470557721us_comparison.jpg

def show_projected_points(image_path, sift_path_not_classified, sift_path_classified, output_image_path):
    red = (0, 0, 255)
    green = (0, 255, 0)

    image = cv2.imread(image_path)

    keypoints_xy_descs_not_classified = np.loadtxt(sift_path_not_classified)
    keypoints_xy_not_classified = keypoints_xy_descs_not_classified[:, 0:2]

    keypoints_xy_descs_classified = np.loadtxt(sift_path_classified)
    keypoints_xy_classified = keypoints_xy_descs_classified[:, 0:2]

    for i in range(int(len(keypoints_xy_descs_not_classified))):
        x = int(keypoints_xy_not_classified[i][0])
        y = int(keypoints_xy_not_classified[i][1])
        center = (x, y)
        cv2.circle(image, center, 3, red, -1)

    for i in range(int(len(keypoints_xy_descs_classified))):
        x = int(keypoints_xy_classified[i][0])
        y = int(keypoints_xy_classified[i][1])
        center = (x, y)
        cv2.circle(image, center, 3, green, -1)

    cv2.imwrite(output_image_path, image)

image_path = sys.argv[1]
sift_path_not_classified = sys.argv[2]
sift_path_classified = sys.argv[3]
output_image_path = sys.argv[4]

show_projected_points(image_path, sift_path_not_classified, sift_path_classified, output_image_path)