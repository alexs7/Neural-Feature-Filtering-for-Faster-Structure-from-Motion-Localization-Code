import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from database import COLMAPDatabase
from parameters import Parameters
from point3D_loader import read_points3d_default, index_dict_reverse
from query_image import read_images_binary, get_image_decs

# Run this file for all datasets, CMU, Retail, LaMAR - binary classifier, visibility and per_image score regressors
# General Notes: you are using conda tf env

# Run this in order:
# 1 - create_nf_training_data.py
# 2 - train_for_nf.py

# TODO: review the below - not relevant anymore
# 4 - prepare_comparison_data.py
# 5 - model_evaluator.py
# 6 - print_eval_NN_results.py
# 7 - under plots/ plots.py and plot_performance_over_percentages_ml.py (or plot_performance_over_percentage_10_only_ml.py)

def get_point3D_score(points3D_scores, current_point3D_id, points3D_id_index):
    point_index = points3D_id_index[current_point3D_id]
    point_score = points3D_scores[0, point_index]
    return point_score

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

    print(" Total Training Size: " + str(all_classes.shape[0]))
    positive_percentage = np.where(all_classes == 1)[0].shape[0] * 100 / all_classes.shape[0]
    negative_percentage = np.where(all_classes == 0)[0].shape[0] * 100 / all_classes.shape[0]
    print(" Positive Percentage %: " + str(positive_percentage))
    print(" Negative Percentage %: " + str(negative_percentage))
    return negative_percentage, positive_percentage

def create_all_data_opencv_sift(params):
    image_path = params.mnm_all_universal_images_path # from create_universal_models.py

    # need to save the opencv db under ML_data
    ml_db_dir = os.path.join(params.base_path, "ML_data/")
    mnm_ml_db_path = os.path.join(ml_db_dir, "ml_database_all_opencv_sift.db")
    # db that training data will be inserted into
    print("Creating database...")
    mnm_ml_db = COLMAPDatabase.create_db_for_all_data_opencv(mnm_ml_db_path) #returns a connection, and drop the table if it exists

    # This database already exists, and it was altered by create_universal_models.py
    mnm_db_path = os.path.join(params.gt_db_path_mnm)
    # db that training data will be read from (connect to it)
    mnm_db = COLMAPDatabase.connect(mnm_db_path)

    print("Loading data...")
    # These are the model info from opencv gt model, created by create_universal_models.py
    opencv_gt_model_images = read_images_binary(params.gt_model_images_path_mnm)
    opencv_gt_model_points3D = read_points3d_default(params.gt_model_points3D_path_mnm)

    # These are the localised images in the Opencv gt model, created by create_universal_models.py
    mnm_gt_localised_images = np.loadtxt(parameters.query_gt_localised_images_txt_path_mnm, dtype=str)

    mnm_ml_db.execute("BEGIN")
    for img_id , img_data in tqdm(opencv_gt_model_images.items()):

        # skip gt images are we do NOT want to train on them
        if(img_data.name in mnm_gt_localised_images):
            continue

        # this happens only in LaMAR and in the base model provided by the authors. Some images have no xys, no points3D, but a pose and db descs.
        # So probably they manually added them. I just skip them. The descs are there because of me, from exmaps file, get_lamar_data.py
        if (img_data.xys.shape[0] == 0):
            continue
        descs = get_image_decs(mnm_db, img_id)
        assert(img_data.xys.shape[0] == img_data.point3D_ids.shape[0] == descs.shape[0]) # just for my sanity

        # read image for rgb values
        img_file = cv2.imread(os.path.join(image_path, img_data.name))

        for i in range(img_data.point3D_ids.shape[0]): # can loop through descs or img_data.xys - same thing
            current_point3D_id = img_data.point3D_ids[i]

            if(current_point3D_id == -1): # means feature (or keypoint) is unmatched
                matched = 0
                xyz = np.array([0, 0, 0])  # safe to use as no image point will ever match to 0,0,0
            else:
                matched = 1
                # point2D_idxs are zero-based indices of the image points that observe the 3D point
                # i is the index of the current image point from the image that observes a point. so i must be in point2D_idxs
                # point2D_idxs are not from the same image, but from all images that observe the same 3D point
                # and correspoinds to image_ids of the point3D
                assert i in opencv_gt_model_points3D[current_point3D_id].point2D_idxs
                xyz = opencv_gt_model_points3D[current_point3D_id].xyz  # np.float64

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

            mnm_ml_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          (img_id,) + (img_name,) + (COLMAPDatabase.array_to_blob(desc),) +
                          (COLMAPDatabase.array_to_blob(xyz),) + (COLMAPDatabase.array_to_blob(xy),) +
                          (int(blue),) + (int(green),) + (int(red),) + (matched,))
    print('Done!')
    mnm_ml_db.commit()

    print("Generating Data Info...")
    all_data = mnm_ml_db.execute("SELECT * FROM data ORDER BY image_id DESC").fetchall()

    all_classes = (row[8] for row in all_data)  # binary values
    all_classes = np.array(list(all_classes))

    print(" Total Training Size: " + str(all_classes.shape[0]))
    positive_percentage = np.where(all_classes == 1)[0].shape[0] * 100 / all_classes.shape[0]
    negative_percentage = np.where(all_classes == 0)[0].shape[0] * 100 / all_classes.shape[0]
    print(" Negative Percentage %: " + str(negative_percentage))
    print(" Positive Percentage %: " + str(positive_percentage))
    return negative_percentage, positive_percentage

def create_data(params, image_path, dataset, use_opencv_sift_models=False):
    if(use_opencv_sift_models):
        print("Generating data for ML.. (using OpenCV SIFT)")
        ml_db_dir = os.path.join(params.base_path, "ML_data/")
        os.makedirs(ml_db_dir, exist_ok=True)
        # This method will read data generated from create_universal_models.py, that used OpenCV SIFT
        # No score here, you might need to add them later if you train a regressor
        # To run the method below you need to first run create_universal_models.py for the dataset you want
        negative_percentage, positive_percentage = create_all_data_opencv_sift(params)
        return negative_percentage, positive_percentage
    else:
        # In this case it will use COLMAP SIFT
        # The base and live data here are used to differentiate between the live and base nad gt images.
        db_live = COLMAPDatabase.connect(params.live_db_path)
        print("Getting pre data for ML..")
        base_model_images = read_images_binary(params.base_model_images_path)
        live_model_images = read_images_binary(params.live_model_images_path)
        live_model_points3D = read_points3d_default(params.live_model_points3D_path)
        # Getting the scores
        points3D_per_session_scores = np.load(params.per_session_decay_scores_path)
        points3D_per_image_scores = np.load(params.per_image_decay_scores_path)
        points3D_visibility_scores = np.load(params.binary_visibility_scores_path)
        points3D_id_index = index_dict_reverse(live_model_points3D)
        # neural filtering data
        ml_db_dir = os.path.join(params.base_path, "ML_data/")
        os.makedirs(ml_db_dir, exist_ok=True)
        ml_db_path = os.path.join(ml_db_dir, "ml_database_all.db")
        print("Creating all training data..")
        negative_percentage, positive_percentage = create_all_data(ml_db_path, live_model_points3D, points3D_id_index,
                                                                   points3D_per_image_scores, points3D_per_session_scores, points3D_visibility_scores,
                                                                   live_model_images, base_model_images, db_live,
                                                                   image_path, params.base_path, dataset)
        return negative_percentage, positive_percentage
    pass

root_path = "/media/iNicosiaData/engd_data/"
dataset = sys.argv[1] #HGE, CAB, LIN (or Other for CMU, retail shop)
use_opencv_sift_models = (sys.argv[2] == '1')

if(use_opencv_sift_models):
    print("Using OpenCV SIFT models")

if(dataset == "HGE" or dataset == "CAB" or dataset == "LIN"):
    base_path = os.path.join(root_path, "lamar", f"{dataset}_colmap_model")
    print("Base path: " + base_path)
    parameters = Parameters(base_path)
    image_path = os.path.join(root_path, "lamar/", dataset)
    negative_percentage, positive_percentage = create_data(parameters, image_path, dataset, use_opencv_sift_models=use_opencv_sift_models)
    np.savetxt(os.path.join(parameters.base_path, "ML_data/", f"{dataset}_classes_percentage.txt"), [negative_percentage, positive_percentage])

if(dataset == "CMU"):
    slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                    "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    for slice_name in slices_names:
        base_path = os.path.join(root_path, "cmu", f"{slice_name}", "exmaps_data")
        print("Base path: " + base_path)
        parameters = Parameters(base_path)
        image_path = os.path.join(base_path, "live", "images")
        negative_percentage, positive_percentage = create_data(parameters, image_path, dataset, use_opencv_sift_models=use_opencv_sift_models)
        np.savetxt(os.path.join(parameters.base_path, "ML_data/", f"CMU_{slice_name}_classes_percentage.txt"), [negative_percentage, positive_percentage])

if(dataset == "RetailShop"):
    base_path = os.path.join(root_path, "retail_shop", "slice1")
    print("Base path: " + base_path)
    parameters = Parameters(base_path)
    image_path = os.path.join(base_path, "live", "images")
    negative_percentage, positive_percentage = create_data(parameters, image_path, dataset, use_opencv_sift_models=use_opencv_sift_models)
    np.savetxt(os.path.join(parameters.base_path, "ML_data/", f"{dataset}_classes_percentage.txt"), [negative_percentage, positive_percentage])
