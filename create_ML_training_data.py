import pdb
import sys
from collections import defaultdict
import numpy as np
from database import COLMAPDatabase
from parameters import Parameters
from point3D_loader import read_points3d_default, index_dict_reverse
from query_image import read_images_binary

def get_image_decs(db, image_id):
    data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(image_id) + "'")
    data = COLMAPDatabase.blob_to_array(data.fetchone()[0], np.uint8)
    descs_rows = int(np.shape(data)[0] / 128)
    descs = data.reshape([descs_rows, 128])  # descs for the whole image
    return descs

def create_training_data(points3D, points3D_id_index, points3D_scores, images_bin, db):
    training_data = np.empty([0, 129])
    save_idx = 0
    cached_image_descs = defaultdict(lambda: np.array([]))

    for k,v in points3D.items():
        point_id = v.id
        point_index = points3D_id_index[point_id]
        point_score = points3D_scores[point_index]
        points_image_ids = points3D[point_id].image_ids #COLMAP adds the image twice some times.
        print("Doing point " + str(point_index + 1) + "/" + str(len(points3D.items())) + ". This point is viewed from " + str(len(points_image_ids)) + " images. Training Data saves: " + str(save_idx), end="\r")
        # Loop through the points' image ids and check if it is seen by any image_ids
        # If it is seen then get the desc for each id.
        for img_id_idx in range(len(points_image_ids)):
            training_data_row = np.zeros([1, 129])
            id = points_image_ids[img_id_idx] #id here is image_id
            descs = np.array([])
            if(cached_image_descs[id].size == 0):
                cached_image_descs[id] = get_image_decs(db, id)
                descs = cached_image_descs[id]
            else:
                descs = cached_image_descs[id]
            assert(descs.size !=0 )
            keypoint_index = points3D[point_id].point2D_idxs[img_id_idx]
            desc = descs[keypoint_index] #keypoints and descs are ordered the same (so I use the point2D_idxs to index descs, it makes sense as the only way to get the SIFT desc is from the db)
            desc = desc.reshape(1, 128) #this is the desc of keypoint with index, keypoint_index, from image with id, id.
            # desc = desc / desc.sum()
            training_data_row[0, 0:128] = desc
            training_data_row[0, 128] = point_score
            training_data = np.r_[training_data, training_data_row]

            if(training_data.shape[0] >= 5000):
                save_idx += 1
                np.save("/home/alex/fullpipeline/colmap_data/Coop_data/slice1/live_model_training_data/training_data_"+str(save_idx)+".npy", training_data)
                training_data = np.empty([0, 129])

    return training_data

base_path = sys.argv[1] # example: "/home/alex/fullpipeline/colmap_data/CMU_data/slice2/" #trailing "/"
parameters = Parameters(base_path)

db_live = COLMAPDatabase.connect(parameters.live_db_path)

live_model_images = read_images_binary(parameters.live_model_images_path)
live_model_points3D = read_points3d_default(parameters.live_model_points3D_path)

points3D_per_image_decay_scores = np.load(parameters.per_image_decay_matrix_path)
points3D_per_image_decay_scores = points3D_per_image_decay_scores.sum(axis=0)
points3D_id_index = index_dict_reverse(live_model_points3D)

training_data = create_training_data(live_model_points3D, points3D_id_index, points3D_per_image_decay_scores, live_model_images, db_live)
np.save("/home/alex/fullpipeline/colmap_data/Coop_data/slice1/live_model_training_data/training_data_last.npy", training_data)
