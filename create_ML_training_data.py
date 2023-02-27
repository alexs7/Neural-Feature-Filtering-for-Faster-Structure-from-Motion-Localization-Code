import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from database import COLMAPDatabase
from parameters import Parameters
from point3D_loader import read_points3d_default, index_dict_reverse
from query_image import read_images_binary, get_image_decs

# The ML part for my second publication starts from here:
# Run this file for all datasets, CMU, Retail, LaMAR - binary classifier, visibility and per_image score regressors
# General Notes: you are using conda tf env

# Run this in order:
# 1 - create_ML_training_data.py
# 2 - train_classification_NF.py

# TODO: review the below - not relevant anymore
# 4 - prepare_comparison_data.py
# 5 - model_evaluator.py
# 6 - print_eval_NN_results.py
# 7 - under plots/ plots.py and plot_performance_over_percentages_ml.py (or plot_performance_over_percentage_10_only_ml.py)

def get_point3D_score(points3D_scores, current_point3D_id, points3D_id_index):
    point_index = points3D_id_index[current_point3D_id]
    point_score = points3D_scores[0, point_index]
    return point_score

def get_input_pre_data_for_ml(params):
    base_model_images = read_images_binary(params.base_model_images_path)
    live_model_images = read_images_binary(params.live_model_images_path)
    live_model_points3D = read_points3d_default(params.live_model_points3D_path)
    # Getting the scores
    points3D_per_session_scores = np.load(params.per_session_decay_scores_path)
    points3D_per_image_scores = np.load(params.per_image_decay_scores_path)
    points3D_visibility_scores = np.load(params.binary_visibility_scores_path)
    points3D_id_index = index_dict_reverse(live_model_points3D)
    return base_model_images, live_model_images, live_model_points3D, points3D_per_session_scores, points3D_per_image_scores, points3D_visibility_scores, points3D_id_index

def create_all_data(ml_db_path, points3D, points3D_id_index,
                    points3D_per_image_scores, points3D_per_session_vals, points3D_visibility_vals,
                    live_images, base_images, db,
                    image_path, base_path, dataset):
    ml_db = COLMAPDatabase.create_db_for_all_data(ml_db_path) #returns a connection, and drop the table if it exists
    ml_db.execute("BEGIN")
    for img_id , img_data in tqdm(live_images.items()):
        # this happens only in LaMAR and in the base model provided by the authors. Some images have no xys, no points3D, but a pose and db descs.
        # So probably they manually added them. I just skip them. The descs are there because of me, from exmaps file, get_lamar_data.py
        if (img_data.xys.shape[0] == 0):
            continue
        descs = get_image_decs(db, img_id)
        assert(img_data.xys.shape[0] == img_data.point3D_ids.shape[0] == descs.shape[0]) # just for my sanity

        # read image for rgb values
        if(dataset == "HGE" or dataset == "CAB" or dataset == "LIN"):
            if (img_id in base_images.keys()):
                img_file = cv2.imread(os.path.join(os.path.join(image_path,"sessions/map/raw_data"), img_data.name))
            if (img_id in live_images.keys() and img_id not in base_images.keys()):
                img_file = cv2.imread(os.path.join(os.path.join(image_path,"sessions/query_phone/raw_data"), img_data.name))  # image_path defaults to live images
        else: #below is for CMU and Retail Shop only
            if(img_id in base_images.keys()):
                img_file = cv2.imread(os.path.join(base_path, "base", "images", img_data.name))
            if (img_id in live_images.keys() and img_id not in base_images.keys()):
                img_file = cv2.imread(os.path.join(image_path, img_data.name)) # image_path defaults to live images

        for i in range(img_data.point3D_ids.shape[0]): # can loop through descs or img_data.xys - same thing
            current_point3D_id = img_data.point3D_ids[i]

            if(current_point3D_id == -1): # means feature (or keypoint) is unmatched
                per_image_score = 0
                per_session_score = 0
                visibility_score = 0
                matched = 0
                xyz = np.array([0, 0, 0])  # safe to use as no image point will ever match to 0,0,0
            else:
                per_image_score = get_point3D_score(points3D_per_image_scores, current_point3D_id, points3D_id_index)
                per_session_score = get_point3D_score(points3D_per_session_vals, current_point3D_id, points3D_id_index)
                visibility_score = get_point3D_score(points3D_visibility_vals, current_point3D_id, points3D_id_index)
                matched = 1
                xyz = points3D[current_point3D_id].xyz  # np.float64

            desc = descs[i] # np.uint8
            xy = img_data.xys[i] #np.float64, same as xyz
            img_name = img_data.name
            y = np.round(xy[1]).astype(int)
            x = np.round(xy[0]).astype(int)
            if(y >= img_file.shape[0] or x >= img_file.shape[1]):
                continue
            brg = img_file[y, x] #opencv conventions
            blue = brg[0]
            green = brg[1]
            red = brg[2]

            ml_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (img_id,) + (img_name,) + (COLMAPDatabase.array_to_blob(desc),) +
                          (per_image_score,) + (per_session_score,) + (visibility_score,) +
                          (COLMAPDatabase.array_to_blob(xyz),) + (COLMAPDatabase.array_to_blob(xy),) +
                          (int(blue),) + (int(green),) + (int(red),) + (matched,))

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

    all_classes = (row[11] for row in all_data)  # binary values
    all_classes = np.array(list(all_classes))

    print(" Total Training Size: " + str(all_sifts.shape[0]))
    print(" per_image_scores mean: " + str(per_image_scores.mean()))
    print(" per_session_scores mean: " + str(per_session_scores.mean()))
    print(" visibility_scores mean: " + str(visibility_scores.mean()))
    print(" per_image_scores std: " + str(per_image_scores.std()))
    print(" per_session_scores std: " + str(per_session_scores.std()))
    print(" visibility_scores std: " + str(visibility_scores.std()))
    ratio = np.where(all_classes == 1)[0].shape[0] / np.where(all_classes == 0)[0].shape[0]
    print("Ratio of Positives to Negatives Classes: " + str(ratio))

def prepare_data_and_create(params, image_path, base_path, dataset):
    db_live = COLMAPDatabase.connect(params.live_db_path)
    print("Getting pre data for ML..")
    base_model_images, live_model_images, live_model_points3D, points3D_per_session_scores, \
        points3D_per_image_scores, points3D_visibility_scores, points3D_id_index = get_input_pre_data_for_ml(params)
    # neural filtering data
    ml_db_dir = os.path.join(base_path, "ML_data/")
    os.makedirs(ml_db_dir, exist_ok=True)
    ml_db_path = os.path.join(ml_db_dir, "ml_database_all.db")
    print("Creating all training data..")
    create_all_data(ml_db_path, live_model_points3D, points3D_id_index, points3D_per_image_scores, points3D_per_session_scores, points3D_visibility_scores,
                    live_model_images, base_model_images, db_live, image_path, base_path, dataset)
    pass

root_path = "/media/iNicosiaData/engd_data/"
dataset = sys.argv[1] #HGE, CAB, LIN (or Other for CMU, retail shop)

if(dataset == "HGE" or dataset == "CAB" or dataset == "LIN"):
    base_path = os.path.join(root_path, "lamar", f"{dataset}_colmap_model")
    print("Base path: " + base_path)
    parameters = Parameters(base_path)
    image_path = os.path.join(root_path, "lamar/", dataset)
    prepare_data_and_create(parameters, image_path, base_path, dataset)

if(dataset == "CMU"):
    slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                    "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    for slice_name in slices_names:
        base_path = os.path.join(root_path, "cmu", f"{slice_name}", "exmaps_data")
        print("Base path: " + base_path)
        parameters = Parameters(base_path)
        image_path = os.path.join(base_path, "live", "images")
        prepare_data_and_create(parameters, image_path, base_path, dataset)

if(dataset == "RetailShop"):
    base_path = os.path.join(root_path, "retail_shop", "slice1")
    print("Base path: " + base_path)
    parameters = Parameters(base_path)
    image_path = os.path.join(base_path, "live", "images")
    prepare_data_and_create(parameters, image_path, base_path, dataset)
