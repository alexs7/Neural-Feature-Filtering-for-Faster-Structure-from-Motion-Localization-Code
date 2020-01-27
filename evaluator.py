import numpy as np
import cv2
from query_image import get_query_image_global_pose
from point3D_loader import get_points3D
from query_image import get_query_image_id

K = np.loadtxt("matrices/pixel_intrinsics_low_640.txt")

def get_pose_from_correspondences(filename,K):
    correspondences = np.loadtxt(filename)
    (_, rvec, tvec, inliers_direct) = cv2.solvePnPRansac(correspondences[:,2:5], correspondences[:,0:2], K, None, iterationsCount = 500, confidence = 0.99, flags = cv2.SOLVEPNP_EPNP)
    rotM = cv2.Rodrigues(rvec)[0]
    pose = np.c_[rotM, tvec]
    pose = np.r_[pose, [np.array([0, 0, 0, 1])]]
    return pose

def show_projected_points(image_path, K, FP, points3D):
    image = cv2.imread(image_path)
    points = K.dot(FP.dot(points3D.transpose())[0:3,:])
    points = points // points[2,:]
    points = points.transpose()
    for i in range(len(points)):
        x = int(points[i][0])
        y = int(points[i][1])
        center = (x, y)
        cv2.circle(image, center, 4, (0, 0, 255), -1)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow("result", image)
    cv2.waitKey(0)

# get COLMAP pose the 3D points the image(s) are looking at
# from: frame_1579029653869.jpg
# to:   frame_1579029667636.jpg
# images between are:
sequence = [ "1579029653869", "1579029654274", "1579029654999", "1579029655387", "1579029655774", "1579029656146",
             "1579029656535", "1579029656939", "1579029657327", "1579029657732", "1579029658137", "1579029658508",
             "1579029658895", "1579029659283", "1579029659688", "1579029660093", "1579029660482", "1579029660886",
             "1579029661274", "1579029661662", "1579029662050", "1579029662438", "1579029662827", "1579029663235",
             "1579029663620", "1579029664025", "1579029664430", "1579029664834", "1579029665241", "1579029665645",
             "1579029666033", "1579029666437", "1579029666842", "1579029667248", "1579029667636"]

CP_start = get_query_image_global_pose("frame_" + sequence[0] + ".jpg")

LP_start = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/query_data/cameraPose_" + sequence[0] + ".txt")
LP_start_dp_oriented = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/query_data/displayOrientedPose_" + sequence[0] + ".txt")
# FP_start = LP_start.dot(np.linalg.inv(LP_start)).dot(CP_start) # Identity anyway..
#
# image_id_start = get_query_image_id("frame_"+first_frame+".jpg")
# points3D_start = get_points3D(image_id_start)
#
# np.savetxt("points3D_start.txt", points3D_start)
#
# show_projected_points("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/current_query_image/frame_" + first_frame + ".jpg", K, FP_start, points3D_start)

colmap_poses = np.empty([4,4,len(sequence)])
print("Getting ground truth poses from colmap..")
for i in range(len(sequence)):
    colmap_pose = get_query_image_global_pose("frame_"+sequence[i]+".jpg")
    colmap_poses[:,:,i] = colmap_pose

relative_pose = np.empty([4, 4])
relative_pose_dp_oriented = np.empty([4, 4])
print("Saving poses..")
for i in range(1 , len(sequence)):

    LP_dp_oriented = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/query_data/displayOrientedPose_" + sequence[i] + ".txt")

    R1_dp_oriented = LP_start_dp_oriented[0:3,0:3]
    t1_dp_oriented = LP_start_dp_oriented[0:3, 3]
    R2_dp_oriented = LP_dp_oriented[0:3,0:3]
    t2_dp_oriented = LP_dp_oriented[0:3,3]
    R1to2_dp_oriented = np.linalg.inv(R2_dp_oriented).dot(R1_dp_oriented)
    T1to2_dp_oriented = np.linalg.inv(R2_dp_oriented).dot((t1_dp_oriented - t2_dp_oriented))

    relative_pose_dp_oriented[0:3, 0:3] = R1to2_dp_oriented
    relative_pose_dp_oriented[0:3, 3] = T1to2_dp_oriented
    relative_pose_dp_oriented[3, :] = [0, 0, 0, 1]

    FP_dp_oriented = relative_pose_dp_oriented.dot(CP_start)

    LP = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/query_data/cameraPose_" + sequence[i] + ".txt")

    R1 = LP_start[0:3,0:3]
    t1 = LP_start[0:3, 3]
    R2 = LP[0:3,0:3]
    t2 = LP[0:3,3]
    R1to2 = np.linalg.inv(R2).dot(R1)
    T1to2 = np.linalg.inv(R2).dot((t1 - t2))

    relative_pose[0:3, 0:3] = R1to2
    relative_pose[0:3, 3] = T1to2
    relative_pose[3, :] = [0, 0, 0, 1]

    FP = relative_pose.dot(CP_start)

    # show_projected_points("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/current_query_image/frame_" + sequence[i] +".jpg", K, FP, points3D_start)

    np.savetxt("final_poses/cameraPoses_" + sequence[i] + ".txt", FP)
    np.savetxt("cameraPose_dp_oriented/cameraPose_dp_oriented" + sequence[i] + ".txt", FP_dp_oriented)
    np.savetxt("colmap_poses/colmap_pose_" + sequence[i] + ".txt", colmap_poses[: , : , i])




