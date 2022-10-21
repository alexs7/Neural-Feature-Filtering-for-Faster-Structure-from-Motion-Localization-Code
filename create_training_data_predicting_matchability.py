# This file will aim to create the data for the RF model from Predicting Matchability - PM (2014) paper
# It will overwrite existing data (db).
# Note: https://stackoverflow.com/questions/5189997/python-db-api-fetchone-vs-fetchmany-vs-fetchall
# Note: https://stackoverflow.com/questions/41464752/git-rebase-interactive-the-last-n-commits
# To create data for all datasets:
# You need to run this file separately as the Neighbours params differ for each dataset - you need to try diff ones.
# so far the neighb. numbers are 13,9,16,35, for CMU in order, and 13 for coop

import os
import sys
from database import COLMAPDatabase
from database import pair_id_to_image_ids
import numpy as np
from random import sample, choice
from tqdm import tqdm
from parameters import Parameters
from query_image import read_images_binary, get_descriptors

def get_matched_decs_from_pairs(pair_ids_image_ids, live_images, db_live):
    all_descs = np.empty([0,129])
    # example: [pair_id: 377957122070 , image_ids: (176.0, 198)]
    for pair_id, image_ids in pair_ids_image_ids.items():
        pair_data = db_live.execute("SELECT rows, data FROM matches WHERE pair_id = " + "'" + str(pair_id) + "'").fetchone()
        rows = pair_data[0]
        cols = 2
        zero_based_indices = COLMAPDatabase.blob_to_array(pair_data[1], np.uint32).reshape([rows, cols])
        image_id_left = image_ids[0]
        image_id_right = image_ids[1]
        _,_, descs_left = get_descriptors(db_live, str(image_id_left))
        _,_, descs_right = get_descriptors(db_live, str(image_id_right))

        image_left_points3D_ids = live_images[image_id_left].point3D_ids #same order as keypoints
        image_right_points3D_ids = live_images[image_id_right].point3D_ids #same order as keypoints

        # only use the matched ones
        # i.e descs that are matched with other descs
        # then the descs/kps that have a point3D is -1, are negative
        # I had to do it this way as if I use all the descs is returns extreme
        # unbalanced data.
        descs_left = np.c_[descs_left[zero_based_indices[:,0]], np.zeros([zero_based_indices[:,0].shape[0],1])]
        descs_right = np.c_[descs_right[zero_based_indices[:,1]], np.zeros([zero_based_indices[:,1].shape[0],1])]
        # the points3D ids that the descs/keypoints are associated with (-1 means no point3D association)
        image_left_points3D_ids = image_left_points3D_ids[zero_based_indices[:,0]]
        image_right_points3D_ids = image_right_points3D_ids[zero_based_indices[:,1]]
        # these indices below will be used to set the matched/unmatched value to the descs
        matched_left_descs_with_points3D_idx = np.where(image_left_points3D_ids != -1)[0]
        matched_right_descs_with_points3D_idx = np.where(image_right_points3D_ids != -1)[0]
        descs_left[matched_left_descs_with_points3D_idx, 128] = 1
        descs_right[matched_right_descs_with_points3D_idx, 128] = 1

        all_descs = np.r_[all_descs, descs_left]
        all_descs = np.r_[all_descs, descs_right]

    return all_descs

def createDataForPredictingMatchabilityComparison(total_neighbours, live_images, db_live_path, db_PM_path):
    print("Creating data..")
    # 21/10/2022 The plan is to use the same methods as in the NNs. Matched feature will be considered one that has a 3D point
    # can use random pairs to get the 3D point info matched/no mathed for each feature
    training_data_db = COLMAPDatabase.create_db_predicting_matchability_data(os.path.join(db_PM_path, "training_data.db"))
    db_live = COLMAPDatabase.connect(db_live_path)

    training_data_db.execute("BEGIN")

    # each tuple_neighbours contains a set of backward_neighbours and forward_neighbours
    for tuple_neighbours in tqdm(total_neighbours):
        backward_neighbours = tuple_neighbours[0]
        forward_neighbours = tuple_neighbours[1]

        pair_ids_b = list(backward_neighbours.keys())
        image_ids_b = list((pair_id_to_image_ids(el) for el in pair_ids_b))
        pair_ids_f = list(forward_neighbours.keys())
        image_ids_f = list((pair_id_to_image_ids(el) for el in pair_ids_f))

        pair_ids = pair_ids_b + pair_ids_f
        image_ids = image_ids_b + image_ids_f
        pair_ids_image_ids = dict(zip(pair_ids, image_ids))

        # for a set of backward_neighbours and forward_neighbours (if 13 each, then 26 total)
        training_descs = get_matched_decs_from_pairs(pair_ids_image_ids, live_images, db_live)

        for training_desc in training_descs:
            desc = training_desc[0:128].astype(np.uint8)
            matched = training_desc[128]
            training_data_db.execute("INSERT INTO data VALUES (?, ?)", (COLMAPDatabase.array_to_blob(desc),) + (matched,))

    print("Committing..")
    training_data_db.commit()
    print("Generating Stats..")
    stats = training_data_db.execute("SELECT * FROM data").fetchall()
    matched = training_data_db.execute("SELECT * FROM data WHERE matched = 1").fetchall()
    unmatched = training_data_db.execute("SELECT * FROM data WHERE matched = 0").fetchall()

    print("Total descs: " + str(len(stats)))
    print("Total matched descs: " + str(len(matched)))
    print("Total unmatched descs: " + str(len(unmatched)))
    print("% of matched decs: " + str(len(matched) * 100 / len(stats)))

    np.savetxt(os.path.join(output_path, "matched_PM.txt"), [len(matched)])
    np.savetxt(os.path.join(output_path, "unmatched_PM.txt"), [len(unmatched)])

    print("Done!")

def get_Neighbours(db_live_path, neighbours_rand_limit):
    # use 64 for neighbours_rand_limit at first then made it in a variable
    db_live = COLMAPDatabase.connect(db_live_path)
    print("Getting the neighbours (from pairs)..")
    pair_ids = db_live.execute("SELECT pair_id FROM matches").fetchall()
    random_starting_pairs_64 = []
    max_neighbours = 13
    pbar = tqdm(total=neighbours_rand_limit)
    final_set_of_neighbours = []
    while len(random_starting_pairs_64) != neighbours_rand_limit:
        rnd_id = choice(pair_ids)[0]
        already_checked = rnd_id in random_starting_pairs_64
        if (already_checked):
            continue
        # as per paper discard pairs with less than 50 matches
        rows = db_live.execute("SELECT rows FROM matches WHERE pair_id = " + "'" + str(rnd_id) + "'").fetchone()
        if (rows[0] < 50):
            continue
        # how many neighbours does the pair have ?
        forward_neighbours = {}
        backward_neighbours = {}
        neighbours_no = 0
        # the loop below will fetch up to max_neighbours for each side
        for i in range(1, max_neighbours + 1):
            forward_id = rnd_id + i
            forward_neighbours_rows = db_live.execute("SELECT rows FROM matches WHERE pair_id = " + "'" + str(forward_id) + "'").fetchone()
            backward_id = rnd_id - i
            backward_neighbours_rows = db_live.execute("SELECT rows FROM matches WHERE pair_id = " + "'" + str(backward_id) + "'").fetchone()
            # 'rows' can be None (if pair_id record doesn't exist) or 0 if not None - fuck me
            if ((backward_neighbours_rows != None and forward_neighbours_rows != None) and
                    (backward_neighbours_rows[0] >= 50 and forward_neighbours_rows[0] >= 50)): #50 is from paper
                forward_neighbours[forward_id] = forward_neighbours_rows
                backward_neighbours[backward_id] = backward_neighbours_rows
                neighbours_no += 1

        # Did we find less than the minimum required neighbours ?
        if (neighbours_no < min_neighbours_no):  # it is below the minimum accepted so choose another pair
            continue

        random_starting_pairs_64.append(rnd_id)  # only add pair here the healthy one, let it check unhealthy ones again.
        # at this point we found a healthy pair id
        final_set_of_neighbours.append((backward_neighbours,forward_neighbours))
        pbar.update(1)

    pbar.close()
    return final_set_of_neighbours

base_path = sys.argv[1]
neighbours_rand_limit = int(sys.argv[2]) # more sample images to get min_neighbours_no from
min_neighbours_no = int(sys.argv[3]) #set to 13 to be super strict, or 0 to accept all

print("Base path: " + base_path)
parameters = Parameters(base_path)
live_model_images = read_images_binary(parameters.live_model_images_path)

db_live_path = os.path.join(base_path, "live/database.db")
output_path = os.path.join(base_path, "predicting_matchability_comparison_data")
os.makedirs(output_path, exist_ok = True)
#for 64 random pairs - 26/10/2022, changed to variable: neighbours_rand_limit
total_neighbours = get_Neighbours(db_live_path, neighbours_rand_limit)
createDataForPredictingMatchabilityComparison(total_neighbours, live_model_images, db_live_path, output_path)
