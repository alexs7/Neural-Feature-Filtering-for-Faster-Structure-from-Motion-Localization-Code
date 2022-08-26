# This file will aim to create the data for the RF model from Predicting Matchability - PM (2014) paper
# then data.py, which is the file that fetches the data for training models
# It will overwrite existing data (db).
# Note: https://stackoverflow.com/questions/5189997/python-db-api-fetchone-vs-fetchmany-vs-fetchall
# Note: https://stackoverflow.com/questions/41464752/git-rebase-interactive-the-last-n-commits
# To create data for all datasets:
# You need to run this file separately as the Neighbours params differ for each - you need to try diff ones.

import os
import sys
from database import COLMAPDatabase
from database import pair_id_to_image_ids
import numpy as np
from random import sample, choice
from tqdm import tqdm

def getImgMatchedUnMathced(zero_based_indices, data):
    descs_rows = data[0]
    descs = COLMAPDatabase.blob_to_array(data[1], np.uint8)
    descs = descs.reshape([descs_rows, 128])  # descs for the whole image
    matched_descs = descs[zero_based_indices]
    unmatched_descs = np.delete(descs, zero_based_indices, axis=0)
    return matched_descs, unmatched_descs

def createDataForPredictingMatchabilityComparison(total_neighbours, db_live_path, db_PM_path):
    print("Creating data..")
    training_data_db = COLMAPDatabase.create_db_predicting_matchability_data(os.path.join(db_PM_path, "training_data.db"))
    db_live = COLMAPDatabase.connect(db_live_path)

    training_data_db.execute("BEGIN")
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

        for pair_id, image_ids in pair_ids_image_ids.items():
            pair_data = db_live.execute("SELECT rows, data FROM matches WHERE pair_id = " + "'" + str(pair_id) + "'").fetchone()
            rows = pair_data[0]
            cols = 2
            zero_based_indices = COLMAPDatabase.blob_to_array(pair_data[1], np.uint32).reshape([rows, cols])
            zero_based_indices_left = zero_based_indices[:,0]
            zero_based_indices_right = zero_based_indices[:,1]
            image_id_left = image_ids[0]
            image_id_right = image_ids[1]
            data_left = db_live.execute("SELECT rows, data FROM descriptors WHERE image_id = " + "'" + str(image_id_left) + "'").fetchone()
            data_right = db_live.execute("SELECT rows, data FROM descriptors WHERE image_id = " + "'" + str(image_id_right) + "'").fetchone()

            img_left_desc_matched, img_left_desc_unmatched = getImgMatchedUnMathced(zero_based_indices_left, data_left)
            img_right_desc_matched, img_right_desc_unmatched = getImgMatchedUnMathced(zero_based_indices_right, data_right)

            # matched
            for pos_sample in img_left_desc_matched:
                training_data_db.execute("INSERT INTO data VALUES (?, ?, ?)",
                                         (image_id_left,) + (COLMAPDatabase.array_to_blob(pos_sample),) + (1,))
            for pos_sample in img_right_desc_matched:
                training_data_db.execute("INSERT INTO data VALUES (?, ?, ?)",
                                         (image_id_left,) + (COLMAPDatabase.array_to_blob(pos_sample),) + (1,))
            # unmatched
            for neg_sample in img_left_desc_unmatched:
                training_data_db.execute("INSERT INTO data VALUES (?, ?, ?)",
                                         (image_id_right,) + (COLMAPDatabase.array_to_blob(neg_sample),) + (0,))
            for neg_sample in img_right_desc_unmatched:
                training_data_db.execute("INSERT INTO data VALUES (?, ?, ?)",
                                         (image_id_right,) + (COLMAPDatabase.array_to_blob(neg_sample),) + (0,))

    print("Committing..")
    training_data_db.commit()
    print("Generating Stats..")
    stats = training_data_db.execute("SELECT * FROM data").fetchall()
    matched = training_data_db.execute("SELECT * FROM data WHERE matched = 1").fetchall()
    unmatched = training_data_db.execute("SELECT * FROM data WHERE matched = 0").fetchall()

    print("Total descs: " + str(len(stats)))
    print("Total matched descs: " + str(len(matched)))
    print("Total unmatched descs: " + str(len(unmatched)))
    print("% of matched decs: " + str(len(matched) * 100 / len(unmatched)))

    np.savetxt(os.path.join(output_path, "matched_PM.txt"), [len(matched)])
    np.savetxt(os.path.join(output_path, "unmatched_PM.txt"), [len(unmatched)])

    print("Done!")

def get_Neighbours(db_live_path):
    db_live = COLMAPDatabase.connect(db_live_path)
    print("Getting the neighbours (from pairs)..")
    pair_ids = db_live.execute("SELECT pair_id FROM matches").fetchall()
    random_starting_pairs_64 = []
    max_neighbours = 13
    pbar = tqdm(total=64)
    final_set_of_neighbours = []
    while len(random_starting_pairs_64) != 64:
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
min_neighbours_no = int(sys.argv[2]) #set to 13 to be super strict, or 0 to accept all

print("Base path: " + base_path)
db_live_path = os.path.join(base_path, "live/database.db")
output_path = os.path.join(base_path, "predicting_matchability_comparison_data")
os.makedirs(output_path, exist_ok = True)
#for 64 random pairs
total_neighbours = get_Neighbours(db_live_path)
createDataForPredictingMatchabilityComparison(total_neighbours, db_live_path, output_path)
