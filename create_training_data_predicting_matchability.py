# This file will aim to create the data for the RF model from Predicting Matchability - PM (2014) paper
# It will overwrite existing PM training data (db).
# Note: https://stackoverflow.com/questions/5189997/python-db-api-fetchone-vs-fetchmany-vs-fetchall
# Note: https://stackoverflow.com/questions/41464752/git-rebase-interactive-the-last-n-commits
# 03/11/2022: The official code (random forest) has been compiled,  g++ -O3 -DNDEBUG -Wall -march=native -o rforest rforest.cpp -I. -lboost_serialization -lboost_iostreams
# This file will also produce the training data for the rforest.cpp
# After this file runs, then compile the tool again: g++ -O3 -DNDEBUG -Wall -march=native -o rforest rforest.cpp -I. -lboost_serialization -lboost_iostreams
# just for sanity and train it: ./rforest -t 25 -d 25 -p pos.txt -n neg.txt -f rforest.gz

# To summarise:
# 1 - Run this file to generate the data for your sklearn model (.db) and the original tool rforest (.txt)
# 2 - Run "train_for_predicting_matchability.py"
# 3 - Run "test_all_models_on_3D_gt_data.py"

# NOTE 14/02/2023:
# Removed the original tool as I got same results with python code.

import os
import sys
from database import COLMAPDatabase
from database import pair_id_to_image_ids, image_ids_to_pair_id
import numpy as np
from tqdm import tqdm
from parameters import Parameters
from query_image import get_descriptors

# def save_to_files_for_original_tool(all_descs, original_tool_path, file_identifier):
#     with open(os.path.join(f"{original_tool_path}", f"pos_{file_identifier}.txt"), 'w') as f:
#         for desc in tqdm(all_descs[np.where(all_descs[:,128] == 1)[0]]):
#             row = ' '.join([str(num) for num in desc[0:128].astype(np.uint8)])
#             f.write(f"{row}\n")
#     with open(os.path.join(f"{original_tool_path}", f"neg_{file_identifier}.txt"), 'w') as f:
#         for desc in tqdm(all_descs[np.where(all_descs[:,128] == 0)[0]]):
#             row = ' '.join([str(num) for num in desc[0:128].astype(np.uint8)])
#             f.write(f"{row}\n")

def get_matched_and_unmatched_descs(c, m, db_live, side=None):
    assert side != None
    pair_id = image_ids_to_pair_id(int(c), int(m))
    pair_data = db_live.execute("SELECT rows, cols, data FROM two_view_geometries WHERE pair_id = " + "'" + str(pair_id) + "'").fetchone()
    rows = pair_data[0]
    cols = pair_data[1]
    data = pair_data[2]
    zero_based_indices = COLMAPDatabase.blob_to_array(data, np.uint32).reshape([rows, cols])

    zero_based_matched_indices_left = zero_based_indices[:, 0]
    zero_based_matched_indices_right = zero_based_indices[:, 1]
    _, _, descs_left = get_descriptors(db_live, str(int(c)))
    _, _, descs_right = get_descriptors(db_live, str(int(m)))
    zero_based_unmatched_indices_left = np.delete(np.arange(descs_left.shape[0]), zero_based_matched_indices_left, axis=0)
    zero_based_unmatched_indices_right = np.delete(np.arange(descs_right.shape[0]), zero_based_matched_indices_right, axis=0)

    if(side=='left'): #central image
        matched = descs_left[zero_based_matched_indices_left]
        unmatched = descs_left[zero_based_unmatched_indices_left]
        matched = np.c_[matched, np.ones([matched[:, 0].shape[0], 1])]
        unmatched = np.c_[unmatched, np.zeros([unmatched[:, 0].shape[0], 1])]
        return matched, unmatched, zero_based_matched_indices_left, zero_based_unmatched_indices_left
    if (side == 'right'): #matched image or right image or m
        matched = descs_right[zero_based_matched_indices_right]
        unmatched = descs_right[zero_based_unmatched_indices_right]
        matched = np.c_[matched, np.ones([matched[:, 0].shape[0], 1])]
        unmatched = np.c_[unmatched, np.zeros([unmatched[:, 0].shape[0], 1])]
        return matched, unmatched, zero_based_matched_indices_right, zero_based_unmatched_indices_right

def gen_training_data(pair_ids, max_neighbours, max_pairs, db_live):
    print("Getting Training Data..")
    image_ids = pair_id_to_image_ids(np.array(pair_ids))
    left_images_ids = image_ids[0] #can also use image_ids[1] here but it's more or less the same

    if(max_pairs == -1): #just use all UNIQUE pair ids
        max_pairs = len(np.unique(left_images_ids.flatten()))

    # TODO: max_pairs might be > than the number of unique left_images_ids images, so we need to check (rare case)
    rand_images = np.random.choice(np.unique(left_images_ids.flatten()), size=max_pairs, replace=False) #pick 500 random images (or central images) unique

    matched_images = {} #this will contain a random image id and its 13 * 2 (26),
    # This is where the "neighbours" are retrieved
    for rand_img_id in tqdm(rand_images):
        # get the neighbour pairs' image_ids indexes
        # to explain, we get the indexes of the random image id from the left/central image of the left and right images id list.
        # example: if rand_img_id == 4322, then we get the indexes of 4322 from left_images_ids or image_ids[0] and then use them to get
        # their individual pair(s) from image_ids[1]. we call those neighbours / matches_ids (below)
        # example: 4322 can be seen in index position 44 and 900. We pick the image_ids from the right images id list (image_ids[1]) at position 44, and 900.
        matches_ids = image_ids[1][np.where(left_images_ids == rand_img_id)[0]].flatten() #the images ids that are a match with rand_img_id
        # picking 13 neighbours here that are continuous makes no sense
        # so we pick 13 * 2 random images ids that are a match to rand_img_id
        if(len(matches_ids) > max_neighbours * 2):
            match_neighbours = np.random.choice(matches_ids, max_neighbours * 2, replace=False)
        else:
            match_neighbours = matches_ids #pick all, at this point it will be less than 26

        if(len(match_neighbours) == 0): #rand_img_id has no matches at all...sad..
            matched_images[rand_img_id] = 0
            continue

        assert (len(match_neighbours) != 0)
        matched_images[rand_img_id] = match_neighbours

    assert(len(matched_images) == max_pairs)

    print(f"Chose {len(matched_images.keys())} samples, or central images.")
    total_neighbours = 0
    all_ms = []
    for c, ms in tqdm(matched_images.items()):
        for m in ms:
            all_ms.append(m)
        total_neighbours += len(ms)
    print(f"An average of {int(total_neighbours/len(matched_images.keys()))} \"neighbours\" (both back and forward), or matched images, per central image (no: {len(matched_images)}).")
    print(f"A total of unique {np.unique(all_ms).shape[0]} \"neighbours\" (both back and forward), or matched images.")

    # At this point I will assign to each image id an array of positive / matched zero-based descs indices
    print("Getting the positive and negative desc indices for each image ..")
    # this contains all images, central and matches and their unique descs indexes
    tracked_indices = {}
    # Do central ones first
    for c, ms in tqdm(matched_images.items()):
        # for each c we only check with the first match, ms[0], can do other but fuck it same thing
        matched, unmatched, matched_indices, unmatched_indices = get_matched_and_unmatched_descs(c, ms[0], db_live, side='left')
        assert (matched.shape[0] == matched_indices.shape[0])
        assert (unmatched.shape[0] == unmatched_indices.shape[0])
        tracked_indices[c] = matched_indices #c will always be unique as matched_images.keys() only contains unique ids

    assert (len(tracked_indices) == max_pairs)

    # Do matched ones second
    first_seen_matched_images = 0
    for c, ms in tqdm(matched_images.items()):
        for m in ms:
            # a right image (m) is an image that forms a pair with c (central image)
            # we have many c ones (unique) and many m ones. some c's can have the same m's.
            # so it is bound to happen that we will see some m again and again.
            # so make sure to keep their descs that were deemed positive once and not replace them with a negative one
            # as per the paper, "All points that appear in at least one match form the positive class"
            matched, unmatched, matched_indices, unmatched_indices = get_matched_and_unmatched_descs(c, m, db_live, side='right')
            assert(matched.shape[0] == matched_indices.shape[0])
            assert(unmatched.shape[0] == unmatched_indices.shape[0])

            if(m in tracked_indices):
                # add new matched_indices and remove duplicates
                tracked_indices[m] = np.unique(np.concatenate((tracked_indices[m], matched_indices), 0))
            else:
                first_seen_matched_images +=1
                tracked_indices[m] = matched_indices

    # tracked_indices will contain less neighbours as it already contains central images ids (max_pairs should be the same length as centrals)
    assert (len(tracked_indices) == max_pairs + first_seen_matched_images)

    # at this point for each central and matched image we have the indices of the positive descs
    # we can easily get the negative ones

    # calculate the size of the descs array
    print("Calculating the size of, all_descs ..")
    decs_no = 0
    for image_id, matched_indices_array in tqdm(tracked_indices.items()):
        _, _, descs = get_descriptors(db_live, str(int(image_id)))
        decs_no += descs.shape[0]
    all_descs = np.empty([decs_no, 129])

    # at this point get the descs using the indices and save them to all_descs
    print("Filling, all_descs ..")
    i = 0
    for image_id, matched_indices_array in tqdm(tracked_indices.items()):
        _, _, descs = get_descriptors(db_live, str(int(image_id)))

        matched = descs[matched_indices_array]
        unmatched = np.delete(descs, matched_indices_array, axis=0)
        matched = np.c_[matched, np.ones([matched[:, 0].shape[0], 1])]
        unmatched = np.c_[unmatched, np.zeros([unmatched[:, 0].shape[0], 1])]

        assert((matched.shape[0] + unmatched.shape[0]) == descs.shape[0])

        for desc in matched:
            all_descs[i] = desc
            i += 1
        for desc in unmatched:
            all_descs[i] = desc
            i += 1

    assert i == all_descs.shape[0]
    return all_descs

def createDataForPredictingMatchabilityComparison(db_live_path, db_PM_path, max_pairs, max_neighbours=13):
    file_identifier = f"{max_pairs}_samples"
    training_data_db_path = os.path.join(db_PM_path, f"training_data_{file_identifier}.db")
    print(f"Creating training data db at {training_data_db_path}.. (will drop table if exists)")
    training_data_db = COLMAPDatabase.create_db_predicting_matchability_data(training_data_db_path)
    db_live = COLMAPDatabase.connect(db_live_path)
    print("Selecting pair_ids..")
    pair_ids = db_live.execute("SELECT pair_id FROM two_view_geometries WHERE rows != 0 ORDER BY rows DESC").fetchall()

    training_data_db.execute("BEGIN")
    training_descs = gen_training_data(pair_ids, max_neighbours, max_pairs, db_live)

    print("Inserting data..")
    for training_desc in tqdm(training_descs):
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

    training_data_db.close()
    print("Done!")

root_path = "/media/iNicosiaData/engd_data/"
dataset = sys.argv[1] #HGE, CAB, LIN (or Other for CMU, retail shop)
max_pairs = int(sys.argv[2]) # more sample images to get "neighbours" from

if(dataset == "HGE" or dataset == "CAB" or dataset == "LIN"):
    base_path = os.path.join(root_path, "lamar", f"{dataset}_colmap_model")
    print("Base path: " + base_path)
    parameters = Parameters(base_path)
    db_live_path = parameters.live_db_path
    output_path = os.path.join(base_path, "predicting_matchability_comparison_data")
    os.makedirs(output_path, exist_ok=True)
    createDataForPredictingMatchabilityComparison(db_live_path, output_path, max_pairs)

if(dataset == "CMU"):
    slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                    "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    for slice_name in slices_names:
        base_path = os.path.join(root_path, "cmu", f"{slice_name}", "exmaps_data")
        print("Base path: " + base_path)
        parameters = Parameters(base_path)
        db_live_path = parameters.live_db_path
        output_path = os.path.join(base_path, "predicting_matchability_comparison_data")
        os.makedirs(output_path, exist_ok=True)
        createDataForPredictingMatchabilityComparison(db_live_path, output_path, max_pairs)

if(dataset == "RetailShop"):
    base_path = os.path.join(root_path, "retail_shop", "slice1")
    print("Base path: " + base_path)
    parameters = Parameters(base_path)
    db_live_path = parameters.live_db_path
    output_path = os.path.join(base_path, "predicting_matchability_comparison_data")
    os.makedirs(output_path, exist_ok=True)
    createDataForPredictingMatchabilityComparison(db_live_path, output_path, max_pairs)


