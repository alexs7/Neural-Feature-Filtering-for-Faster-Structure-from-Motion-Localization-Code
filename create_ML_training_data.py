import pdb
import sys
from collections import defaultdict
import numpy as np
from database import COLMAPDatabase
from parameters import Parameters
from point3D_loader import read_points3d_default, index_dict_reverse
from query_image import read_images_binary

# The ML part for my second publication starts from here:
# This file will use data from previous publication (live model + all sessions) and the live database. You can run this on your alienware or ogg/weatherwax
# You run these in order:
# (Note, load the python venv: source venv/bin/activate (not in docker!))
# run, create_ML_training_data.py (see below)
# then run any model such as regression.py, regression_rf.py, using docker on weatherwax or ogg cs.bath.ac.uk.
# (Note the docker command to run is: hare run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -v "$(pwd)":/fullpipeline --workdir /fullpipeline -e COLUMNS="`tput cols`" -e LINES="`tput lines`" -ti bath:2020-gpu
# (Note, you will need docker to run the models because it uses gpus, the venv uses python3.6 for some reason)
# then run, view_ML_model_results.py, to evaluate the model on unseen data!
# then run, create_ML_visualization_data.py, to create data from unseen images to evaluate visually the models!

# Command example (for coop data, paths might change):
# python3 create_ML_training_data.py /home/alex/fullpipeline/colmap_data/Coop_data/slice1/
#                                    /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/ml_database.db

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

def create_training_data(ml_db, points3D, points3D_id_index, points3D_scores, images, db):
    img_index = 0
    for img_id , img_data in images.items():
        print("Doing image " + str(img_index + 1) + "/" + str(len(images.items())), end="\r")
        descs = get_image_decs(db, img_id)
        assert(img_data.xys.shape[0] == img_data.point3D_ids.shape[0] == descs.shape[0]) # just for my sanity
        for i in range(img_data.point3D_ids.shape[0]): # can loop through descs or img_data.xys - same thing
            current_point3D_id = img_data.point3D_ids[i]

            if(current_point3D_id == -1): # means feature is unmatched
                score = -99.0
                matched = 0
            else:
                score = get_point3D_score(points3D_scores, current_point3D_id, points3D_id_index)
                matched = 1

            if (current_point3D_id == -1):
                xyz = np.array([0, 0, 0]) # safe to use as no image point will ever match to 0,0,0
            else:
                xyz = points3D[current_point3D_id].xyz #np.float64

            desc = descs[i] # np.uint8
            xy = img_data.xys[i] #np.float64, same as xyz
            img_name = img_data.name

            ml_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (img_id,) + (img_name,) + (COLMAPDatabase.array_to_blob(desc),) + (score,) +
                              (COLMAPDatabase.array_to_blob(xyz),) + (COLMAPDatabase.array_to_blob(xy),) + (matched,))
        img_index +=1
    print()
    print('Done')
    ml_db.commit()

base_path = sys.argv[1] # example: "/home/alex/fullpipeline/colmap_data/CMU_data/slice1/" #trailing "/"
parameters = Parameters(base_path)

db_live = COLMAPDatabase.connect(parameters.live_db_path)

live_model_images = read_images_binary(parameters.live_model_images_path)
live_model_points3D = read_points3d_default(parameters.live_model_points3D_path)

points3D_per_image_decay_scores = np.load(parameters.per_image_decay_matrix_path)
points3D_per_image_decay_scores = points3D_per_image_decay_scores.sum(axis=0)
points3D_id_index = index_dict_reverse(live_model_points3D)

# i.e /home/alex/fullpipeline/colmap_data/alfa_mega/slice1/ML_data/database.db / or ml_database.db / or coop/alfa_mega
# make sure you delete the database (.db) file first!
ml_db_path = sys.argv[2]
ml_data_db = COLMAPDatabase.create_connection(ml_db_path)
training_data = create_training_data(ml_data_db,
                                     live_model_points3D,
                                     points3D_id_index,
                                     points3D_per_image_decay_scores,
                                     live_model_images,
                                     db_live)
