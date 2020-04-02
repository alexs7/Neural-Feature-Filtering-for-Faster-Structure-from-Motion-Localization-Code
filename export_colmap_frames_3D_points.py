import numpy as np
import cv2
from query_image import read_images_binary
from point3D_loader import read_points3d_binary_id
from scipy.spatial.transform import Rotation as R

K = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/matrices/pixel_intrinsics_low_640_landscape.txt")

path_images = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/26_03_2020/fresh_uni_large/model_run_1/model/0/images.bin"
path_points = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/26_03_2020/fresh_uni_large/model_run_1/model/0/points3D.bin"

images = read_images_binary(path_images)
k = 0

for k,v in images.items():

    # colmap_to_arcore_matrix = np.array([
    #     [1, 0, 0, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, -1, 0]])
    #
    # pose_r = v.qvec2rotmat()
    # pose_t = v.tvec
    # pose = np.c_[pose_r, pose_t]
    # pose = np.r_[pose, [np.array([0, 0, 0, 1])]]
    #
    # pose = colmap_to_arcore_matrix.dot(pose)
    #
    # pose_r = R.from_dcm(pose[0:3,0:3]).as_quat()
    # pose_t = pose[0:3,3]
    # pose = np.r_[pose_r, pose_t]

    pose_r = v.qvec2rotmat()
    pose_t = v.tvec
    pose = np.c_[pose_r, pose_t]
    pose = np.r_[pose, [np.array([0, 0, 0, 1])]]

    rot = np.array(pose[0:3,0:3])
    cam_center = -rot.transpose().dot(pose_t)

    pose_r = v.qvec
    pose = np.r_[pose_r, cam_center]

    np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/26_03_2020/fresh_uni_large/model_run_1/pose_"+str(k)+".txt", pose)

    points3D = read_points3d_binary_id(path_points,images[k].id)
    np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/26_03_2020/fresh_uni_large/model_run_1/points3D_"+str(k)+".txt", points3D)

    image = cv2.imread("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/26_03_2020/fresh_uni_large/run_1/"+v.name)
    points = v.xys
    for i in range(len(points)):
        x = int(points[i][0])
        y = int(points[i][1])
        center = (x, y)
        cv2.circle(image, center, 4, (0, 255, 0), -1)
    cv2.imwrite("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/26_03_2020/fresh_uni_large/frame_projected_"+str(k)+".jpg", image)

    k = k + 1