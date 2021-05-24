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
# (Note the docker command to run is (and before run this "hare reserve 20000" to reserve a port , change to "fullpipeline dir first" ):
# hare run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1 -v "$(pwd)":/fullpipeline --workdir /fullpipeline -e COLUMNS="`tput cols`" -e LINES="`tput lines`" -p 20000:80 -ti bath:2020-gpu
# (Note, you will need docker to run the models because it uses gpus, the venv uses python3.6 for some reason - it's ok I think)
# then run, view_ML_model_results.py, to evaluate the model on unseen data!
# then run, create_ML_visualization_data.py, to create data from unseen images to evaluate visually the models!

# for docker you might also need to run these for "cv2" and "cvxpnpl"
# apt-get update && apt-get install ffmpeg libsm6 libxext6 libblas-dev liblapack-dev -y && pip install opencv-contrib-python && pip install scs && pip install cvxpnpl

# Tensorboard Notes:
# https://chadrick-kwag.net/how-to-manually-write-to-tensorboard-from-tf-keras-callback-useful-trick-when-writing-a-handful-of-validation-metrics-at-once/
# You need 2 terminals
# 1 - to run tensorboard, you ssh with "ssh -L 9999:localhost:20000 ar2056@weatherwax.cs.bath.ac.uk", and run this too "source ~/venv_basic/bin/activate"
# then run "tensorboard --logdir colmap_data/Coop_data/slice1/ML_data/results/ --port 20000" inside fullpipeline/ (might need to reserve a port with hare)
# the you visit "http://localhost:9999" on your local machine.
# 2 - the terminal you usually run the hare command from above and Tensorflow will read from the dir you store the results.
# Will need to flush with the writer though, https://stackoverflow.com/questions/52483296/when-do-i-have-to-use-tensorflows-filewriter-flush-method (doesn't work...)

# Docker Notes:
# create an image using the docker file under "/homes/ar2056/docker"
# hare build -t ar2056/basic . (whatever name you prefer)
# hare run -dit --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --workdir /home -v /homes/ar2056/:/home/ --name ar2056_basic ar2056/basic:latest (build container)
# run this first (from laptop), then launch Pycharm for remote dev
# ssh -L 6000:172.17.0.5:22 ar2056@weatherwax.cs.bath.ac.uk (make sure the IP points to a full-working docker, and the docker has to have the ip of 172.17.0.5 or same)
# use git locally on your laptop - the cloud does not like git
# get IP of container with "hare inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container name>"

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

def prepare_data_for_training(db_path_all, db_path_train, db_path_test, test_size = 0.1, shuffle = True, random_state = 42):
    db_all = COLMAPDatabase.connect_ML_db(db_path_all)
    # "_class" = classification
    # "_reg" = regression
    db_train = COLMAPDatabase.create_db_for_training_data(db_path_train)
    db_test = COLMAPDatabase.create_db_for_test_data(db_path_test)

    all_data = db_all.execute("SELECT sift, score, matched FROM data ORDER BY image_id DESC").fetchall()

    # begin transactions
    db_train.execute("BEGIN")
    db_test.execute("BEGIN")

    all_sifts = (COLMAPDatabase.blob_to_array(row[0], np.uint8) for row in all_data)
    all_sifts = np.array(list(all_sifts))

    all_scores = (row[1] for row in all_data)  # continuous values
    all_scores = np.array(list(all_scores))

    all_targets = (row[2] for row in all_data)  # binary values
    all_targets = np.array(list(all_targets))

    indices = np.arange(len(all_data))

    print("Splitting data into test/train..")
    # classification
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(all_sifts, all_targets, indices, test_size=test_size, shuffle=shuffle, random_state=random_state)

    print("Data Info...")
    print(" Total Training Size: " + str(idx_train.shape[0]))
    print(" Total Test Size: " + str(idx_test.shape[0]))
    print(" values_train mean: " + str(all_scores[idx_train].mean())) #negative mean because of -99 values
    print(" values_test mean: " + str(all_scores[idx_test].mean()))
    print(" classes_train mean: " + str(all_targets[idx_train].mean()))
    print(" classes_test mean: " + str(all_targets[idx_test].mean()))

    for i in range(len(idx_train)):
        curr_index = idx_train[i]
        print("Inserting entry " + str(i) + "/" + str(X_train.shape[0]), end="\r")
        db_train.execute("INSERT INTO data VALUES (?, ?, ?)", (COLMAPDatabase.array_to_blob(all_sifts[curr_index,:]),) + (all_scores[curr_index],) + (int(all_targets[curr_index]),))
    for i in range(len(idx_test)):
        curr_index = idx_test[i]
        print("Inserting entry " + str(i) + "/" + str(X_test.shape[0]), end="\r")
        db_test.execute("INSERT INTO data VALUES (?, ?, ?)", (COLMAPDatabase.array_to_blob(all_sifts[curr_index,:]),) + (all_scores[curr_index],) + (int(all_targets[curr_index]),))

    print()
    print("Committing..")

    db_train.execute("COMMIT")
    db_test.execute("COMMIT")
    print("Done!")

    return

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

def create_all_data(ml_db_path, points3D, points3D_id_index, points3D_scores, images, db):
    ml_db = COLMAPDatabase.create_db_for_all_data(ml_db_path) #returns a connection
    img_index = 0
    ml_db.execute("BEGIN")
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

points3D_per_image_decay_scores = np.load(parameters.per_image_decay_matrix_path) #switch this to session scores if needed
points3D_per_image_decay_scores = points3D_per_image_decay_scores.sum(axis=0)
points3D_id_index = index_dict_reverse(live_model_points3D)

# i.e /home/alex/fullpipeline/colmap_data/alfa_mega/slice1/ML_data/database.db / or ml_database.db / or coop/alfa_mega
# make sure you delete the databases (.db) file first!
ml_db_path = sys.argv[2] #colmap_data/Coop_data/slice1/ML_data/ml_database_all.db
db_path_train = sys.argv[3] #colmap_data/Coop_data/slice1/ML_data/ml_database_train.db
db_path_test = sys.argv[4] #colmap_data/Coop_data/slice1/ML_data/ml_database_test.db

print("Creating all data..")
create_all_data(ml_db_path, live_model_points3D, points3D_id_index, points3D_per_image_decay_scores, live_model_images, db_live)
print("Preparing all data for training..")
prepare_data_for_training(ml_db_path, db_path_train, db_path_test, test_size = 0.1, shuffle = True, random_state = 42)
