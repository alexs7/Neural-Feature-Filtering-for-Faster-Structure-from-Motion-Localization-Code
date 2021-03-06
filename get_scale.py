# This file is just to get scales between COLMAP and ARCORE, or COLMAP and COLMAP.
# this file was copied from the newer branch (as I got the scale using another way before submission), with some minor changes
# 02/07/2021: Also check this as a new way to get the scale: https://youtu.be/sgLOU7vyz3g?t=258
import glob
import sys

import numpy as np
import random
from query_image import read_images_binary, get_image_camera_center_by_name, get_images_names

def calc_scale_COLMAP_ARCORE(arcore_devices_poses_path, colmap_model_images_path):

    model_images = read_images_binary(colmap_model_images_path)
    model_images_names_temp = get_images_names(model_images)

    ar_core_cam_centers = {} #This in metric
    ar_core_poses_names = []
    for file in glob.glob(arcore_devices_poses_path+"displayOrientedPose_*.txt"):
        pose = np.loadtxt(file)
        cam_center = np.array(pose[0:3, 3]) # remember in ARCore the matrices' t component is the camera center in the world
        ar_core_poses_name = "frame_"+file.split("_")[-1].split(".")[0]+".jpg"
        ar_core_poses_names.append(ar_core_poses_name)
        ar_core_cam_centers[ar_core_poses_name] = cam_center

    model_images_names = []
    sessions_names = {}
    for image_name in model_images_names_temp: #filtering stage
        if len(image_name.split('/')) == 2: #dealing with names such as 'session_9/frame_1592760573956.jpg'
            full_name = image_name
            image_name = full_name.split('/')[1]
            sessions_names[image_name] = full_name.split('/')[0]
        if image_name in ar_core_poses_names:
            model_images_names.append(image_name)

    # assert (len(model_images_names) == len(ar_core_poses_names)) - trick this will fail! why? because ar_core_poses_names are just the names, model_images_names are the localised one so less (for that arcore session)!
    print("model_images_names size: " + str(len(model_images_names)))

    scales = []
    for i in range(5000): #just to be safe
        random_images = random.sample(model_images_names, 2)

        arcore_1_center = ar_core_cam_centers[random_images[0]]
        arcore_2_center = ar_core_cam_centers[random_images[1]]

        # This is to check if random_images[0]/random_images[1] came from a name such as 'session_9/frame_1592760573956.jpg', so I get their
        # centers using their fullname, i.e 'session_9/frame_1592760573956.jpg'
        sessions_name_1 = ""
        if(sessions_names[random_images[0]]):
            sessions_name_1 = sessions_names[random_images[0]] + "/"

        sessions_name_2 = ""
        if (sessions_names[random_images[1]]):
            sessions_name_2 = sessions_names[random_images[1]] + "/"

        model_cntr1 = get_image_camera_center_by_name(sessions_name_1+random_images[0], model_images)
        model_cntr2 = get_image_camera_center_by_name(sessions_name_2+random_images[1], model_images)

        model_cam_dst = np.linalg.norm(model_cntr1 - model_cntr2)
        arcore_cam_dst = np.linalg.norm(arcore_1_center - arcore_2_center) #in meters

        scale = arcore_cam_dst / model_cam_dst
        scales.append(scale)

    scale = np.mean(scales)

    return scale

# arcore_poses_path = sys.argv[1]
# colmap_poses_path = sys.argv[2]
# calc_scale_COLMAP_ARCORE(arcore_poses_path,colmap_poses_path)
