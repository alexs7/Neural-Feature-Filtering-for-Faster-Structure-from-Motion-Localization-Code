import numpy as np
import cv2
from query_image import get_query_image_global_pose
from point3D_loader import get_points3D
from query_image import get_query_image_id
import glob
import scipy.io as sio

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

def get_ARCore_poses(dir):
    print("Getting ARCore poses..")
    poses = []
    for filename in glob.glob(dir + "/" + 'cameraPose_*.txt'):
        poses.append(np.loadtxt(filename))
    return poses

def get_sequence(dir):
    sequence = []
    for filename in glob.glob(dir+"/"+'cameraPose_*.txt'): #can be anything
        sequence.append(filename.split("_")[-1].split(".")[0])
    return sequence

def get_COLMAP_poses(array):
    print("Getting COLMAP poses..")
    poses = []
    for i in range(len(array)):
        print("At: " + str(round(100 * i / len(array),2)) + "%", end="\r")
        poses.append(get_query_image_global_pose("frame_" + array[i] + ".jpg"))
    return poses

def get_Relative_Poses(local_poses, global_poses):
    relative_poses = []
    gp_start = global_poses[0]
    lp_start = local_poses[0]

    fp = lp_start.dot(np.linalg.inv(lp_start)).dot(gp_start)
    relative_poses.append(fp)

    relative_pose = np.empty([4, 4])
    for i in range(1, len(local_poses)):
        lp =  local_poses[i]

        R1 = lp_start[0:3, 0:3]
        t1 = lp_start[0:3, 3]
        R2 = lp[0:3, 0:3]
        t2 = lp[0:3, 3]
        R1to2 = np.linalg.inv(R2).dot(R1)
        T1to2 = np.linalg.inv(R2).dot((t1 - t2))

        relative_pose[0:3, 0:3] = R1to2
        relative_pose[0:3, 3] = T1to2
        relative_pose[3, :] = [0, 0, 0, 1]

        fp = relative_pose.dot(gp_start)
        relative_poses.append(fp)

    return relative_poses

def save_poses(local_poses, global_poses, relative_poses):
    for i in range(len(relative_poses)):
        np.savetxt("ar_core_poses/pose_" + str(i) + ".txt", local_poses[i])
        np.savetxt("global_poses/pose_" + str(i) + ".txt", global_poses[i])
        np.savetxt("relative_poses/pose_" + str(i) + ".txt", relative_poses[i])


# check and load the files here
# also check the trajectory if it didnt break in ARCore
query_dir = '/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/query_data'
ARCore_poses = get_ARCore_poses(query_dir)
sequence = get_sequence(query_dir)
COLMAP_poses = get_COLMAP_poses(sequence)
relative_Poses = get_Relative_Poses(ARCore_poses, COLMAP_poses)

save_poses(ARCore_poses,COLMAP_poses,relative_Poses)

#
# CP_start = get_query_image_global_pose("frame_"+first_frame+".jpg")
#
# LP_start = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/query_data/cameraPose_" + first_frame + ".txt")
# FP_start = LP_start.dot(np.linalg.inv(LP_start)).dot(CP_start) # Identity anyway..
#
# image_id_start = get_query_image_id("frame_"+first_frame+".jpg")
# points3D_start = get_points3D(image_id_start)
#
# np.savetxt("points3D_start.txt", points3D_start)
#
# show_projected_points("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/current_query_image/frame_" + first_frame + ".jpg", K, FP_start, points3D_start)
#
# colmap_poses = np.empty([4,4,len(sequence)])
# for i in range(len(sequence)):
#     colmap_pose = get_query_image_global_pose("frame_"+sequence[i]+".jpg")
#     colmap_poses[:,:,i] = colmap_pose
#
# relative_pose = np.empty([4, 4])
# for i in range(1 , len(sequence)):
#
#     LP = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/query_data/cameraPose_"+sequence[i]+".txt")
#
#     R1 = LP_start[0:3,0:3]
#     t1 = LP_start[0:3, 3]
#     R2 = LP[0:3,0:3]
#     t2 = LP[0:3,3]
#     R1to2 = np.linalg.inv(R2).dot(R1)
#     T1to2 = np.linalg.inv(R2).dot((t1 - t2))
#
#     relative_pose[0:3, 0:3] = R1to2
#     relative_pose[0:3, 3] = T1to2
#     relative_pose[3, :] = [0, 0, 0, 1]
#
#     FP = relative_pose.dot(CP_start)
#
#     show_projected_points("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/current_query_image/frame_" + sequence[i] +".jpg", K, FP, points3D_start)
#
#     np.savetxt("final_poses/final_pose_" + sequence[i] + ".txt", FP)
#     np.savetxt("colmap_poses/colmap_pose_" + sequence[i] + ".txt", colmap_poses[: , : , i])

# relative_pose = np.empty([4, 4])
# for i in range(1 , len(sequence)):
#
#     FP = sio.loadmat("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/matlab_res/final_pose_"+sequence[i]+".txt.mat")
#     FP = np.array(FP.get('pose_new'))
#     # FP = relative_pose.dot(CP_start)
#
#     show_projected_points("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/current_query_image/frame_" + sequence[i] +".jpg", K, FP, points3D_start)
#
#     # np.savetxt("final_poses/final_pose_" + sequence[i] + ".txt", FP)
#     # np.savetxt("colmap_poses/colmap_pose_" + sequence[i] + ".txt", colmap_poses[: , : , i])


