import pdb
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
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

def get_point3D_score(points3D_scores, current_point3D_id, points3D_id_index):
    point_index = points3D_id_index[current_point3D_id]
    point_score = points3D_scores[point_index]
    return point_score

def create_training_data(base_path, points3D, points3D_id_index, points3D_scores, images, db):
    dims = 134
    training_data = pd.DataFrame()
    img_index = -1
    save_idx = -1
    for img_id , img_data in images.items():
        print("Doing image " + str(img_index + 1) + "/" + str(len(images.items())), end="\r")
        descs = get_image_decs(db, img_id)
        assert(img_data.xys.shape[0] == img_data.point3D_ids.shape[0] == descs.shape[0])
        for i in range(img_data.point3D_ids.shape[0]): # can loop through descs or img_data.xys - same thing
            current_point3D_id = img_data.point3D_ids[i]

            if(current_point3D_id == -1):
                score = 0
            else:
                score = get_point3D_score(points3D_scores, current_point3D_id, points3D_id_index)

            if (current_point3D_id == -1):
                xyz = np.array([0, 0, 0]) # safe to use as no image point will ever match to 0,0,0
            else:
                xyz = points3D[current_point3D_id].xyz

            desc = descs[i]
            xy = img_data.xys[i]
            img_name = img_data.name
            training_data_row = np.zeros([1, dims])
            training_data_row[0, 0:128] = desc
            training_data_row[0, 128] = score
            training_data_row[0, 129:132] = xyz
            training_data_row[0, 132:134] = xy

            df_row = pd.DataFrame({img_name: training_data_row[0]}).transpose()
            training_data = pd.concat([training_data, df_row], axis=0)

            if (training_data.shape[0] >= 2500):
                save_idx += 1
                training_data.to_csv(base_path + "live_model_training_data/training_data_" + str(save_idx) + ".csv")
                training_data = pd.DataFrame()

        img_index +=1
    # save remaining rows - at this point whatever is left just save it
    save_idx += 1
    training_data.to_csv(base_path+"live_model_training_data/training_data_"+str(save_idx)+".csv")

base_path = sys.argv[1] # example: "/home/alex/fullpipeline/colmap_data/CMU_data/slice2/" #trailing "/"
parameters = Parameters(base_path)

db_live = COLMAPDatabase.connect(parameters.live_db_path)

live_model_images = read_images_binary(parameters.live_model_images_path)
live_model_points3D = read_points3d_default(parameters.live_model_points3D_path)

points3D_per_image_decay_scores = np.load(parameters.per_image_decay_matrix_path)
points3D_per_image_decay_scores = points3D_per_image_decay_scores.sum(axis=0)
points3D_id_index = index_dict_reverse(live_model_points3D)

# i.e /home/alex/fullpipeline/colmap_data/alfa_mega/slice1/, the base model with all the sessions localised in it
base_path = sys.argv[1]
training_data = create_training_data(base_path, live_model_points3D, points3D_id_index, points3D_per_image_decay_scores, live_model_images, db_live)
