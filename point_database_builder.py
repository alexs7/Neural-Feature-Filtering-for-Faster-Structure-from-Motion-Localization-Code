import numpy as np
import pdb
import sys
import cv2
import os
import point3D_loader
from colmap_database import COLMAPDatabase

# This script should return a text file with all the 3D points and their
# SIFT average and their id. Each row will be [SIFT average, point id, xyz]

db_path = sys.argv[1]
points3D_txt_path = sys.argv[2]
images_text_file = sys.argv[3]

print("Opening images file..")
f = open(images_text_file, 'r')
lines = f.readlines()
lines = lines[4:] #skip comments
f.close()

print("Loading points 3D..")
points3D = point3D_loader.load_points3D(points3D_txt_path)

db = COLMAPDatabase.connect(db_path)

images_names = db.execute("SELECT name FROM images")
images_names = images_names.fetchall()
all_raw_data = np.empty((0, 129))

for images_name in images_names:

    image_id = db.execute("SELECT image_id FROM images WHERE name = "+"'"+str(images_name[0])+"'")
    image_id = str(image_id.fetchone()[0])
    images_name = str(images_name[0]).split(".")[0]

    print("Getting correspondences for image " + images_name)

    image_keypoints_data = db.execute("SELECT data FROM keypoints WHERE image_id = "+ "'" + image_id + "'")
    image_keypoints_data = image_keypoints_data.fetchone()[0]
    image_keypoints_data_cols = db.execute("SELECT cols FROM keypoints WHERE image_id = "+ "'" + image_id + "'")
    image_keypoints_data_cols = int(image_keypoints_data_cols.fetchone()[0])
    image_keypoints_data = COLMAPDatabase.blob_to_array(image_keypoints_data, np.float32)
    pdb.set_trace()
    image_keypoints_data_rows = int(np.shape(image_keypoints_data)[0]/image_keypoints_data_cols)
    image_keypoints_data = image_keypoints_data.reshape(image_keypoints_data_rows, image_keypoints_data_cols)
    image_keypoints_data_xy = image_keypoints_data[:,0:2]

    image_descriptors_data = db.execute("SELECT data FROM descriptors WHERE image_id = "+ "'" + image_id + "'")
    image_descriptors_data = image_descriptors_data.fetchone()[0]
    image_descriptors_data = COLMAPDatabase.blob_to_array(image_descriptors_data, np.uint8)
    descs_rows = int(np.shape(image_descriptors_data)[0]/128)
    image_descriptors_data = image_descriptors_data.reshape([descs_rows,128])

    for i in range(0,len(lines),2):
        if (lines[i].split(" ")[0] == str(image_id)):
            points2D_x_y_3Did = lines[i+1] # POINTS2D[] as (X, Y, POINT3D_ID)
            break

    pdb.set_trace()
    points2D_x_y_3Did = points2D_x_y_3Did[:-1].split(" ") # -1 for the \n
    points3Dids = np.empty((0, 1))

    points2D_x_y_3Did = np.reshape(points2D_x_y_3Did,[np.shape(points2D_x_y_3Did)[0]/3,3])
    points3Dids = points2D_x_y_3Did[:,2].astype(np.float32)
    points3Dids_rows = np.shape(points3Dids)[0]

    keypoints_xy_descriptors_3DpointId = np.concatenate((keypoints_xy_descriptors, np.reshape(points3Dids,[points3Dids_rows,1])), axis = 1)
    descriptors_3DpointId = keypoints_xy_descriptors_3DpointId[:,2:131]

    all_raw_data = np.concatenate((all_raw_data, descriptors_3DpointId), axis=0)

points3Did_average_xyz = np.empty((0, 132))
print("Averaging...")
for i in range(np.shape(points3D)[0]):
    point3Did = points3D[i,0]
    indices = np.where(all_raw_data[:,128] == point3Did)
    point3D_xyz = points3D[np.where(points3D[:,0] == point3Did)][0][1:4]
    if indices[0].size != 0:
        subset_all_raw_data = all_raw_data[indices][:,0:128]
        mean = np.mean(subset_all_raw_data, axis=0)
        mean = mean.reshape([1,128])
        elem = np.append(mean, point3Did)
        elem = np.append(elem, point3D_xyz)
        elem = elem.astype(np.float64)
        elem = elem.reshape([1,132])
        points3Did_average_xyz = np.concatenate((points3Did_average_xyz, elem), axis=0)

print("Writing to file...")
os.system("mkdir "+data_dir+"/direct_matching_results")
np.savetxt(data_dir+"/direct_matching_results/averages_3Dpoints_xyz.txt", points3Did_average_xyz)