import pdb
import sys
from collections import defaultdict
import numpy as np
from database import COLMAPDatabase
from parameters import Parameters
from point3D_loader import read_points3d_default, index_dict_reverse
from query_image import read_images_binary
from sklearn.model_selection import train_test_split

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
#                                    /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/ml_database_all.db
#                                    /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/ml_database_train_class.db
#                                    /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/ml_database_test_class.db
#                                    /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/ml_database_train_reg.db
#                                    /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/ml_database_test_reg.db
# one liner:
# python3 create_ML_training_data.py /homes/ar2056/fullpipeline/colmap_data/Coop_data/slice1/ /homes/ar2056/fullpipeline/colmap_data/Coop_data/slice1/ML_data/ml_database_all.db /homes/ar2056/fullpipeline/colmap_data/Coop_data/slice1/ML_data/ml_database_train_class.db /homes/ar2056/fullpipeline/colmap_data/Coop_data/slice1/ML_data/ml_database_test_class.db /homes/ar2056/fullpipeline/colmap_data/Coop_data/slice1/ML_data/ml_database_train_reg.db /homes/ar2056/fullpipeline/colmap_data/Coop_data/slice1/ML_data/ml_database_test_reg.db

# Mine not needed
def split_data(features, target, test_percentage, randomize = False):
    if(randomize):
        print("Randomizing data")
        union = np.c_[features, target]
        np.random.shuffle(union)
        features = union[:, 0:128]
        target = union[:, 128]
    rows_no = features.shape[0] #or test , same thing
    train_percentage = 1 - test_percentage
    train_max_idx = int(np.floor(rows_no * train_percentage))
    X_train = features[0 :  train_max_idx , :]
    y_train = target[0 : train_max_idx]
    X_test = features[train_max_idx : , :]
    y_test = target[train_max_idx :]
    return X_train, y_train, X_test, y_test

def prepare_data_for_training(db_path_all, db_path_train_class, db_path_test_class, db_path_train_reg, db_path_test_reg, test_size = 0.1, shuffle = True, random_state = 42):
    db_all = COLMAPDatabase.connect_ML_db(db_path_all)
    # "_class" = classification
    # "_reg" = regression
    db_train_class = COLMAPDatabase.create_db_for_training_data_class(db_path_train_class)
    db_test_class = COLMAPDatabase.create_db_for_test_data_class(db_path_test_class)
    db_train_reg = COLMAPDatabase.create_db_for_training_data_reg(db_path_train_reg)
    db_test_reg = COLMAPDatabase.create_db_for_test_data_reg(db_path_test_reg)

    all_data = db_all.execute("SELECT sift, score, matched FROM data ORDER BY image_id DESC").fetchall()

    all_sifts = (COLMAPDatabase.blob_to_array(row[0], np.uint8) for row in all_data)
    all_sifts = np.array(list(all_sifts))

    all_scores = (row[1] for row in all_data)  # continuous values
    all_scores = np.array(list(all_scores))

    all_targets = (row[2] for row in all_data)  # binary values
    all_targets = np.array(list(all_targets))

    print("Splitting data into test/train..")
    # classification
    X_train, X_test, y_train, y_test = train_test_split(all_sifts, all_targets, test_size=test_size, shuffle=shuffle, random_state=random_state)

    # preproseccing
    # standard scaling - mean normalization
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()

    # min-max normalization - this  might not be needed for binary classification
    # y_train = ( y_train - y_train.min() ) / ( y_train.max() - y_train.min() )
    # y_test = ( y_test - y_test.min() ) / ( y_test.max() - y_test.min() )

    print("Classification Data...")
    print(" Total Training Size: " + str(X_train.shape[0]))
    print(" Total Test Size: " + str(X_test.shape[0]))
    print(" y_train mean: " + str(y_train.mean()))
    print(" y_test mean: " + str(y_test.mean()))

    for i in range(X_train.shape[0]):
        print("Inserting entry " + str(i) + "/" + str(X_train.shape[0]), end="\r")
        db_train_class.execute("INSERT INTO data VALUES (?, ?)", (COLMAPDatabase.array_to_blob(X_train[i,:]),) + (y_train[i],))
        db_train_class.commit()
    for i in range(X_test.shape[0]):
        print("Inserting entry " + str(i) + "/" + str(X_test.shape[0]), end="\r")
        db_test_class.execute("INSERT INTO data VALUES (?, ?)", (COLMAPDatabase.array_to_blob(X_test[i,:]),) + (y_test[i],))
        db_test_class.commit()

    print("Done!")

    print("Splitting data into test/train..")
    # regression
    X_train, X_test, y_train, y_test = train_test_split(all_sifts, all_scores, test_size=test_size, shuffle=shuffle, random_state=random_state)

    # standard scaling - mean normalization
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()

    # min-max normalization
    y_train = ( y_train - y_train.min() ) / ( y_train.max() - y_train.min() )
    y_test = ( y_test - y_test.min() ) / ( y_test.max() - y_test.min() )

    print("Regression Data...")
    print(" Total Training Size: " + str(X_train.shape[0]))
    print(" Total Test Size: " + str(X_test.shape[0]))
    print(" y_train mean: " + str(y_train.mean()))
    print(" y_test mean: " + str(y_test.mean()))

    for i in range(X_train.shape[0]):
        print("Inserting entry " + str(i) + "/" + str(X_train.shape[0]), end="\r")
        db_train_reg.execute("INSERT INTO data VALUES (?, ?)", (COLMAPDatabase.array_to_blob(X_train[i,:]),) + (y_train[i],))
        db_train_reg.commit()
    for i in range(X_test.shape[0]):
        print("Inserting entry " + str(i) + "/" + str(X_test.shape[0]), end="\r")
        db_test_reg.execute("INSERT INTO data VALUES (?, ?)", (COLMAPDatabase.array_to_blob(X_test[i,:]),) + (y_test[i],))
        db_test_reg.commit()

    print("Done!")

    return

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

def create_all_data(ml_db_path, points3D, points3D_id_index, points3D_scores, images, db):
    ml_db = COLMAPDatabase.create_db_for_all_data(ml_db_path)
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
    print('Done!')
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
# make sure you delete the databases (.db) file first!
ml_db_path = sys.argv[2]
db_path_train_class = sys.argv[3]
db_path_test_class = sys.argv[4]
db_path_train_reg = sys.argv[5]
db_path_test_reg = sys.argv[6]

print("Creating all data..")
create_all_data(ml_db_path, live_model_points3D, points3D_id_index, points3D_per_image_decay_scores, live_model_images, db_live)
print("Preparing all data for training..")
prepare_data_for_training(ml_db_path, db_path_train_class, db_path_test_class, db_path_train_reg, db_path_test_reg, test_size = 0.1, shuffle = True, random_state = 42)
