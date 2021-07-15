import os
import pdb
import sys
from collections import defaultdict
import numpy as np
from database import COLMAPDatabase
from parameters import Parameters
from point3D_loader import read_points3d_default, index_dict_reverse
from query_image import read_images_binary
from sklearn.model_selection import train_test_split

# refer to create_ML_training_data.py for a complete overview
# this file is used to create train the network for the network(s) trained on all the CMU data
# command to run: python3 create_ML_training_data.py colmap_data/CMU_data/slice3/ colmap_data/CMU_data/slice4/ colmap_data/CMU_data/slice6/ colmap_data/CMU_data/slice10/ colmap_data/CMU_data/slice11/

def get_image_decs(db, image_id): #not to be confused with get_queryDescriptors() in feature_matching_generator.py - that one normalises descriptors.
    data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(image_id) + "'")
    data = COLMAPDatabase.blob_to_array(data.fetchone()[0], np.uint8)
    descs_rows = int(np.shape(data)[0] / 128)
    descs = data.reshape([descs_rows, 128])  # descs for the whole image
    return descs

def get_point3D_score(points3D_scores, current_point3D_id, points3D_id_index):
    point_index = points3D_id_index[current_point3D_id]
    point_score = points3D_scores[point_index]
    return point_score

def create_all_data(ml_db_path, slices):
    ml_db = COLMAPDatabase.create_db_for_all_data(ml_db_path) #returns a connection

    for slice in slices:
        print("Doing slice: " + slice)
        base_path = slice #just for readability
        parameters = Parameters(base_path)
        db = COLMAPDatabase.connect(parameters.live_db_path)
        images = read_images_binary(parameters.live_model_images_path)
        points3D = read_points3d_default(parameters.live_model_points3D_path)
        # Getting the scores
        points3D_reliability_scores_matrix = np.load(parameters.per_image_decay_matrix_path)
        points3D_heatmap_vals_matrix = np.load(parameters.per_session_decay_matrix_path)
        points3D_visibility_matrix = np.load(parameters.binary_visibility_matrix_path)
        points3D_reliability_scores = points3D_reliability_scores_matrix.sum(axis=0)
        points3D_heatmap_vals = points3D_heatmap_vals_matrix.sum(axis=0)
        points3D_visibility_vals = points3D_visibility_matrix.sum(axis=0)
        points3D_id_index = index_dict_reverse(points3D)

        img_index = 0
        ml_db.execute("BEGIN")
        for img_id , img_data in images.items():
            print("Doing image " + str(img_index + 1) + "/" + str(len(images.items())), end="\r")
            descs = get_image_decs(db, img_id)
            assert(img_data.xys.shape[0] == img_data.point3D_ids.shape[0] == descs.shape[0]) # just for my sanity
            for i in range(img_data.point3D_ids.shape[0]): # can loop through descs or img_data.xys - same thing
                current_point3D_id = img_data.point3D_ids[i]

                if(current_point3D_id == -1): # means feature is unmatched
                    per_image_score = 0
                    per_session_score = 0
                    visibility_score = 0
                    matched = 0
                    xyz = np.array([0, 0, 0])  # safe to use as no image point will ever match to 0,0,0
                else:
                    per_image_score = get_point3D_score(points3D_reliability_scores, current_point3D_id, points3D_id_index)
                    per_session_score = get_point3D_score(points3D_heatmap_vals, current_point3D_id, points3D_id_index)
                    visibility_score = get_point3D_score(points3D_visibility_vals, current_point3D_id, points3D_id_index)
                    matched = 1
                    xyz = points3D[current_point3D_id].xyz  # np.float64

                desc = descs[i] # np.uint8
                xy = img_data.xys[i] #np.float64, same as xyz
                img_name = img_data.name

                ml_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                              (img_id,) + (img_name,) + (COLMAPDatabase.array_to_blob(desc),) +
                              (per_image_score,) + (per_session_score,) + (visibility_score,) +
                              (COLMAPDatabase.array_to_blob(xyz),) + (COLMAPDatabase.array_to_blob(xy),) + (matched,))
            img_index +=1

        print()
        print('Done!')
        ml_db.commit()

        print("Generating Data Info...")
        all_data = ml_db.execute("SELECT * FROM data ORDER BY image_id DESC").fetchall()

        all_sifts = (COLMAPDatabase.blob_to_array(row[2], np.uint8) for row in all_data)
        all_sifts = np.array(list(all_sifts))

        per_image_scores = (row[3] for row in all_data)  # continuous values
        per_image_scores = np.array(list(per_image_scores))

        per_session_scores = (row[4] for row in all_data)  # continuous values
        per_session_scores = np.array(list(per_session_scores))

        visibility_scores = (row[5] for row in all_data)  # continuous values
        visibility_scores = np.array(list(visibility_scores))

        all_classes = (row[8] for row in all_data)  # binary values
        all_classes = np.array(list(all_classes))

        print(" Total Training Size: " + str(all_sifts.shape[0]))
        print(" per_image_scores mean: " + str(per_image_scores.mean()))
        print(" per_session_scores mean: " + str(per_session_scores.mean()))
        print(" visibility_scores mean: " + str(visibility_scores.mean()))
        print(" per_image_scores std: " + str(per_image_scores.std()))
        print(" per_session_scores std: " + str(per_session_scores.std()))
        print(" visibility_scores std: " + str(visibility_scores.std()))
        ratio = np.where(all_classes == 1)[0].shape[0] / np.where(all_classes == 0)[0].shape[0]
        print(" Ratio of Positives to Negatives Classes: " + str(ratio))
        print(" --------------------------->")

ml_db_dir = os.path.join("colmap_data/CMU_data/", "ML_data/")
os.makedirs(ml_db_dir, exist_ok = True)
ml_db_path = os.path.join(ml_db_dir, "ml_database_cmu_all.db")

print("Creating all training data for all CMU slices..")
slices = sys.argv
create_all_data(ml_db_path , slices)
