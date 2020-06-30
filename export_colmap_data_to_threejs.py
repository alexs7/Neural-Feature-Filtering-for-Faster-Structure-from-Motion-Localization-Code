from point3D_loader import read_points3d_default
import numpy as np
import cv2
from query_image import read_images_binary
from point3D_loader import read_points3d_binary_id
import sqlite3
import sys
import os
from database import COLMAPDatabase


def savePoints3DxyzToFile(points3D_xyz):
    np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/points3D.txt", points3D_xyz)

# 28/06/2020 old code might still be useful

# source_3D_points = sys.argv[1] #bin file
# dest_3D_points = sys.argv[2] #text file

# clean up old files
# os.system("rm /Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/*")

# db = COLMAPDatabase.connect("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/database.db")
#
# # 3D points
# print("Exporting 3D Points")
# points3D = read_points3d_default(source_3D_points)

# print("Total Points: " + str(len(points3D)))

# write all 3D points
# print("Writing all 3D points")
# all_points3D =  np.empty([0,4])
# for k,v in points3D.items():
#     data = np.array([v.id , v.xyz[0], v.xyz[1] , v.xyz[2]])
#     data = data.reshape(1,4)
#     all_points3D = np.r_[all_points3D, data]
# np.savetxt(dest_3D_points, all_points3D)

# print("Writing all 3D points - RGB values")
# all_points3D_rgb =  np.empty([0,4])
# for k,v in points3D.items():
#     data = np.array([v.id , v.rgb[0], v.rgb[1] , v.rgb[2]])
#     data = data.reshape(1,4)
#     all_points3D_rgb = np.r_[all_points3D_rgb, data]
# np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/all_rgb_points3D.txt", all_points3D_rgb)
#
# print("Getting observation count")
# observations = []
# for k,v in points3D.items():
#     observations.append(len(v.image_ids))
# observations = np.array(observations)
# observations_mean = observations.mean()
#
# print("Writing all 3D points - observation mean")
# all_points3D_obv_count =  np.empty([0,4])
# for k,v in points3D.items():
#     if(len(v.image_ids) > observations_mean ):
#         data = np.array([v.id, v.xyz[0], v.xyz[1], v.xyz[2]])
#         data = data.reshape(1, 4)
#         all_points3D_obv_count = np.r_[all_points3D_obv_count, data]
# np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/all_xyz_points3D_obvs_mean.txt", all_points3D_obv_count)
#
# print("Getting mean desc for each 3D point") #one file for each
# for k,v in points3D.items():
#     points3D_descs =  np.empty([0,128])
#     for i in range(len(v.image_ids)): #TODO: this might need to be unique? no! because indexing is important here
#         img_id = v.image_ids[i]
#         data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(img_id) + "'")
#         data = blob_to_array(data.fetchone()[0], np.uint8)
#         descs_rows = int(np.shape(data)[0] / 128)
#         descs = data.reshape([descs_rows, 128])
#         desc = descs[v.point2D_idxs[i]]
#         desc = desc.reshape(1, 128)
#         points3D_descs = np.r_[points3D_descs, desc]
#     # np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/desc_for_point3D_" + str(v.id)+".txt", points3D_descs)
#     np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/mean_decs/mean_for_descs_for_point3D_" + str(v.id)+".txt", points3D_descs.mean(axis=0).reshape(1,128))

# Rest of colmap data (poses)
# print("Exporting Rest of Data..")
# # K = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/matrices/pixel_intrinsics_low_640_landscape.txt")
# path_images = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/model/0/images.bin"
# path_points = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/model/0/points3D.bin"
#
# images = read_images_binary(path_images)
#
# for k,v in images.items():
#     pose_r = v.qvec2rotmat()
#     pose_t = v.tvec
#     pose = np.c_[pose_r, pose_t]
#     pose = np.r_[pose, [np.array([0, 0, 0, 1])]]
#
#     rot = np.array(pose[0:3,0:3])
#     cam_center = -rot.transpose().dot(pose_t)
#
#     pose_r = v.qvec
#     pose = np.r_[pose_r, cam_center]
#
#     np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/pose_"+str(k)+".txt", pose)
#
#     # points viewed by image
#     # points3D = read_points3d_binary_id(path_points,images[k].id)
#     # np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/points3D_"+str(k)+".txt", points3D)
#
#     # image = cv2.imread("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/"+v.name)
#     # points = v.xys
#     # for i in range(len(points)):
#     #     x = int(points[i][0])
#     #     y = int(points[i][1])
#     #     center = (x, y)
#     #     cv2.circle(image, center, 4, (0, 255, 0), -1)
#     # cv2.imwrite("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/frame_projected_"+str(k)+".jpg", image)
#     # k = k + 1
#
# print("Getting the query pose..")
# path_images_new_model = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/images.bin"
# images_new = read_images_binary(path_images_new_model)
#
# for k,v in images_new.items():
#     if(v.name == "query.jpg"):
#         pose_r = v.qvec2rotmat()
#         pose_t = v.tvec
#         pose = np.c_[pose_r, pose_t]
#         pose = np.r_[pose, [np.array([0, 0, 0, 1])]]
#
#         rot = np.array(pose[0:3, 0:3])
#         cam_center = -rot.transpose().dot(pose_t)
#
#         pose_r = v.qvec
#         pose = np.r_[pose_r, cam_center]
#         np.savetxt(
#             "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/query_pose.txt", pose)
#
# np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/images_no.txt", [len(images.items())], fmt='%i')
#
