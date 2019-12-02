import sqlite3
import numpy as np
import sys
import pdb
from colmap_database import COLMAPDatabase
from point3d import Point3D

def load_points3D(points3D_text_file):
    points3D = np.empty((0, 4))
    f = open(points3D_text_file, 'r')
    lines = f.readlines()
    lines = lines[3:] #skip comments
    f.close()

    for i in range(len(lines)):
        line = lines[i].split(" ")
        point3Did = line[0]
        point3Did_x = line[1]
        point3Did_y = line[2]
        point3Did_z = line[3]
        points3D = np.append(points3D, [point3Did, point3Did_x, point3Did_y, point3Did_z ])

    points3D = np.reshape(points3D, [int(np.shape(points3D)[0]/4),4])
    points3D = points3D.astype(float)
    return points3D

def parse_points3D(points3D_text_file , db):

    f = open(points3D_text_file, 'r')
    lines = f.readlines()
    lines = lines[3:] #skip comments
    f.close()

    all_points3D = []

    for i in range(len(lines)):
        line = lines[i].split()
        pointId = line[0]
        point3D = Point3D(pointId)

        image_id_points_map = line[8:]

        print("Loading done " + str( round(i * 100 / len(lines)) ) + "%",  end = '\r' )

        for i in range(0, len(image_id_points_map) ,2):
            image_id = image_id_points_map[i]
            point_idx = image_id_points_map[i+1]

            image_keypoints_data = db.execute("SELECT data FROM keypoints WHERE image_id = "+ "'" + image_id + "'")
            image_keypoints_data = image_keypoints_data.fetchone()[0]
            image_keypoints_data_cols = db.execute("SELECT cols FROM keypoints WHERE image_id = "+ "'" + image_id + "'")
            image_keypoints_data_cols = int(image_keypoints_data_cols.fetchone()[0])
            image_keypoints_data = COLMAPDatabase.blob_to_array(image_keypoints_data, np.float32)
            image_keypoints_data_rows = int(np.shape(image_keypoints_data)[0]/image_keypoints_data_cols)
            image_keypoints_data = image_keypoints_data.reshape(image_keypoints_data_rows, image_keypoints_data_cols)
            image_keypoints_data_xy = image_keypoints_data[:,0:2]

            image_descriptors_data = db.execute("SELECT data FROM descriptors WHERE image_id = "+ "'" + image_id + "'")
            image_descriptors_data = image_descriptors_data.fetchone()[0]
            image_descriptors_data = COLMAPDatabase.blob_to_array(image_descriptors_data, np.uint8)
            descs_rows = int(np.shape(image_descriptors_data)[0]/128)
            image_descriptors_data = image_descriptors_data.reshape([descs_rows,128])

            desc = np.reshape(image_descriptors_data[int(point_idx),:], [1,128])
            point3D.add_desc(desc)

        point3D.avg_descs()
        all_points3D.append(point3D)

    return all_points3D

def get_descriptor_array(points3D_objects):
    descs = np.empty((0, 128))
    for i in range(len(points3D_objects)):
        mean_desc = np.reshape(points3D_objects[i].mean_desc, [1,128])
        descs = np.concatenate((descs, mean_desc), axis = 0)
    return descs


























