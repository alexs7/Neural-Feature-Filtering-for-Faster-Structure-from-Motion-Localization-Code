# This file will aim to create data for the RF model from Predicting Matchability (2014) paper
# Note: https://stackoverflow.com/questions/5189997/python-db-api-fetchone-vs-fetchmany-vs-fetchall
# Note: https://stackoverflow.com/questions/41464752/git-rebase-interactive-the-last-n-commits
import os
import sys
from database import COLMAPDatabase
from database import pair_id_to_image_ids
import numpy as np
from random import sample, choice
from tqdm import tqdm

def getMatchedDescsFromPair(zero_based_indices, image_ids, db_live):
    image_1_id = int(image_ids[0])
    image_2_id = int(image_ids[1])

    image_1_rows_descs = db_live.execute("SELECT rows, data FROM descriptors WHERE image_id = " + "'" + str(image_1_id) + "'").fetchone()
    image_1_descs_rows = image_1_rows_descs[0]
    image_1_descs_data = image_1_rows_descs[1]
    image_1_descs = COLMAPDatabase.blob_to_array(image_1_descs_data, np.uint8)
    image_1_descs = image_1_descs.reshape(image_1_descs_rows, 128)

    image_2_rows_descs = db_live.execute("SELECT rows, data FROM descriptors WHERE image_id = " + "'" + str(image_2_id) + "'").fetchone()
    image_2_descs_rows = image_2_rows_descs[0]
    image_2_descs_data = image_2_rows_descs[1]
    image_2_descs = COLMAPDatabase.blob_to_array(image_2_descs_data, np.uint8)
    image_2_descs = image_2_descs.reshape(image_2_descs_rows, 128)

    matched_descs_img1 = image_1_descs[zero_based_indices[:, 0]]
    matched_descs_img2 = image_2_descs[zero_based_indices[:, 1]]

    unmatched_descs_img1 = np.delete(image_1_descs, zero_based_indices[:, 0], axis=0)
    unmatched_descs_img2 = np.delete(image_2_descs, zero_based_indices[:, 1], axis=0)

    assert(image_1_descs.shape[0] == unmatched_descs_img1.shape[0] + matched_descs_img1.shape[0])
    assert(image_2_descs.shape[0] == unmatched_descs_img2.shape[0] + matched_descs_img2.shape[0])

    return matched_descs_img1, unmatched_descs_img1, matched_descs_img2, unmatched_descs_img2

def createDataForPredictingMatchabilityComparison(output_path, db_live_path):
    predicting_matchability_db_path = os.path.join(output_path, "training_data.db")
    training_data_db = COLMAPDatabase.create_db_predicting_matchability_data(predicting_matchability_db_path)
    db_live = COLMAPDatabase.connect(db_live_path)

    print("Getting the pairs..")
    pair_ids = db_live.execute("SELECT pair_id FROM matches").fetchall()
    random_starting_pairs_64 = []
    max_neighbours = 13
    pbar = tqdm(total=64)
    training_data_db.execute("BEGIN")
    while len(random_starting_pairs_64) != 64:
        rnd_id = choice(pair_ids)[0]
        already_checked = rnd_id in random_starting_pairs_64
        if (already_checked):
            continue
        # as per paper discard pairs with less than 50 matches
        rows = db_live.execute("SELECT rows FROM matches WHERE pair_id = " + "'" + str(rnd_id) + "'").fetchone()
        if(rows[0] < 50):
            continue
        # how many neighbours does the pair have ?
        forward_neighbours = {}
        backward_neighbours = {}
        neighbours_no = 0
        for i in range(1, max_neighbours + 1):
            forward_id = rnd_id + i
            forward_neighbours_rows = db_live.execute("SELECT rows FROM matches WHERE pair_id = " + "'" + str(forward_id) + "'").fetchone()
            forward_neighbours[forward_id] = forward_neighbours_rows
            backward_id = rnd_id - i
            backward_neighbours_rows = db_live.execute("SELECT rows FROM matches WHERE pair_id = " + "'" + str(backward_id) + "'").fetchone()
            backward_neighbours[backward_id] = backward_neighbours_rows
            # 'rows' can be None (if pair_id record doesn't exist) or 0 if not None - fuck me
            if((backward_neighbours_rows != None and forward_neighbours_rows != None) and
                    (backward_neighbours_rows[0] > 0 and forward_neighbours_rows [0] > 0)):
                neighbours_no +=1

        # Did we find less than the minimum required neighbours ?
        if(neighbours_no < min_neighbours_no): #it is below the minimum accepted so choose another pair
            continue

        # getting descs from forward_neighbours
        for pair_id, matches_no in forward_neighbours.items():
            image_ids = pair_id_to_image_ids(pair_id)
            pair_data = db_live.execute("SELECT rows, data FROM matches WHERE pair_id = " + "'" + str(pair_id) + "'").fetchone()
            if(pair_data == None or pair_data[0] == 0): #rows = pair_data[0] at this stage
                continue
            rows = pair_data[0]
            cols = 2
            zero_based_indices = COLMAPDatabase.blob_to_array(pair_data[1], np.uint32).reshape([rows, cols])
            pos_descs_img1, neg_descs_img1, pos_descs_img2, neg_descs_img2 = getMatchedDescsFromPair(zero_based_indices, image_ids, db_live)

            # adding 4 rows at a time
            image_ids = np.asarray(image_ids)
            # image_ids.dtype - dtype('float64')  # matched_descs_img1.dtype - dtype('uint8')
            for pos_sample in pos_descs_img1:
                training_data_db.execute("INSERT INTO data VALUES (?, ?, ?)",
                                         (COLMAPDatabase.array_to_blob(image_ids),) + (COLMAPDatabase.array_to_blob(pos_sample),) + (1,))
            for pos_sample in pos_descs_img2:
                training_data_db.execute("INSERT INTO data VALUES (?, ?, ?)",
                                         (COLMAPDatabase.array_to_blob(image_ids),) + (COLMAPDatabase.array_to_blob(pos_sample),) + (1,))
            for neg_sample in neg_descs_img1:
                training_data_db.execute("INSERT INTO data VALUES (?, ?, ?)",
                                         (COLMAPDatabase.array_to_blob(image_ids),) + (COLMAPDatabase.array_to_blob(neg_sample),) + (0,))
            for neg_sample in neg_descs_img2:
                training_data_db.execute("INSERT INTO data VALUES (?, ?, ?)",
                                         (COLMAPDatabase.array_to_blob(image_ids),) + (COLMAPDatabase.array_to_blob(neg_sample),) + (0,))

        # getting descs from backward_neighbours
        for pair_id, matches_no in backward_neighbours.items():
            image_ids = pair_id_to_image_ids(pair_id)
            pair_data = db_live.execute("SELECT rows, data FROM matches WHERE pair_id = " + "'" + str(pair_id) + "'").fetchone()
            if(pair_data == None or pair_data[0] == 0): #rows = pair_data[0] at this stage
                continue
            rows = pair_data[0]
            cols = 2
            zero_based_indices = COLMAPDatabase.blob_to_array(pair_data[1], np.uint32).reshape([rows, cols])
            pos_descs_img1, neg_descs_img1, pos_descs_img2, neg_descs_img2 = getMatchedDescsFromPair(zero_based_indices, image_ids, db_live)

            # adding 4 rows at a time
            image_ids = np.asarray(image_ids)
            # image_ids.dtype - dtype('float64')  # matched_descs_img1.dtype - dtype('uint8')
            for pos_sample in pos_descs_img1:
                training_data_db.execute("INSERT INTO data VALUES (?, ?, ?)", (COLMAPDatabase.array_to_blob(image_ids),) + (COLMAPDatabase.array_to_blob(pos_sample),) + (1,))
            for pos_sample in pos_descs_img2:
                training_data_db.execute("INSERT INTO data VALUES (?, ?, ?)", (COLMAPDatabase.array_to_blob(image_ids),) + (COLMAPDatabase.array_to_blob(pos_sample),) + (1,))
            for neg_sample in neg_descs_img1:
                training_data_db.execute("INSERT INTO data VALUES (?, ?, ?)", (COLMAPDatabase.array_to_blob(image_ids),) + (COLMAPDatabase.array_to_blob(neg_sample),) + (0,))
            for neg_sample in neg_descs_img2:
                training_data_db.execute("INSERT INTO data VALUES (?, ?, ?)", (COLMAPDatabase.array_to_blob(image_ids),) + (COLMAPDatabase.array_to_blob(neg_sample),) + (0,))

        # add this point add the random pair id as we got something from it
        random_starting_pairs_64.append(rnd_id)
        pbar.update(1)
    pbar.close()
    training_data_db.commit()

    print("Generating Stats..")
    stats = training_data_db.execute("SELECT * FROM data").fetchall()
    matched = training_data_db.execute("SELECT * FROM data WHERE matched = 1").fetchall()
    unmatched = training_data_db.execute("SELECT * FROM data WHERE matched = 0").fetchall()

    print("Total descs: " + str(len(stats)))
    print("Total matched descs: " + str(len(matched)))
    print("Total unmatched descs: " + str(len(unmatched)))
    print("Done!")

base_path = sys.argv[1]
min_neighbours_no = int(sys.argv[2]) #set to 13 to be super strict, or 0 to accept all

print("Base path: " + base_path)
db_live_path = os.path.join(base_path, "live/database.db")
output_path = os.path.join(base_path, "predicting_matchability_comparison_data")
os.makedirs(output_path, exist_ok = True)
createDataForPredictingMatchabilityComparison(output_path, db_live_path)
