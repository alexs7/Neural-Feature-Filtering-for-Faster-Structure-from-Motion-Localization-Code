import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from database import COLMAPDatabase
from parameters import Parameters
from point3D_loader import read_points3d_default, index_dict_reverse
from query_image import read_images_binary, get_descriptors, get_keypoints_xy

# Run this file for all datasets, CMU, Retail, LaMAR - binary classifier
# General Notes: you are using conda tf env
# Note: You must create the universal models first, using create_universal_models.py
# Another note: This file generates the training data from the images in the point cloud only, that is the LOCALISED images.
# no point training on addition non localised images, as they will only bring more unmatched data and doesn't align with our training principle which is
# to train on 3D data (a point cloud), i.e. xy - XYZ - matched or not.

# Run this in order:
# 1 - create_universal_models.py
# 2 - create_nf_training_data.py
# 3 - train_for_nf.py

def create_all_data_opencv_sift(params):
    image_path = params.mnm_all_universal_images_path # from create_universal_models.py

    # need to save the opencv db under ML_data
    ml_db_dir = os.path.join(params.base_path, "ML_data/")
    mnm_ml_db_path = os.path.join(ml_db_dir, "ml_database_all_opencv_sift.db")
    # db that training data and testing data will be inserted into
    print("Creating database (might be dropping table if it exists)...")
    mnm_ml_db = COLMAPDatabase.create_db_for_all_data_opencv(mnm_ml_db_path) #returns a connection, and drop the table if it exists

    # This database already exists, and it was altered by create_universal_models.py
    mnm_db_path = os.path.join(params.gt_db_path_mnm)
    # db that training data and testing data will be read from (connect to it)
    mnm_db = COLMAPDatabase.connect(mnm_db_path)

    print("Loading data...")
    # These are the model info from opencv gt model, created by create_universal_models.py
    opencv_gt_model_images = read_images_binary(params.gt_model_images_path_mnm)
    opencv_gt_model_points3D = read_points3d_default(params.gt_model_points3D_path_mnm)

    mnm_ml_db.execute("BEGIN")
    for img_id , img_data in tqdm(opencv_gt_model_images.items()): #using localised images only

        # this happens only in LaMAR and in the base model provided by the authors. Some images have no xys, no points3D, but a pose and db descs.
        # So probably they manually added them. I just skip them. The descs are there because of me, from exmaps file, get_lamar_data.py
        if (img_data.xys.shape[0] == 0):
            print(f"Skipping image {img_data.name} because it has no xys")
            continue

        # get data for image here
        is_image_base = mnm_db.execute("SELECT base FROM images WHERE image_id == ?", (img_id,)).fetchone()[0]
        is_image_live = mnm_db.execute("SELECT live FROM images WHERE image_id == ?", (img_id,)).fetchone()[0]
        is_image_gt = mnm_db.execute("SELECT gt FROM images WHERE image_id == ?", (img_id,)).fetchone()[0]

        rows = mnm_db.execute("SELECT rows FROM keypoints WHERE image_id == ?", (img_id,)).fetchone()[0] #or kps no
        octaves = COLMAPDatabase.blob_to_array(mnm_db.execute("SELECT octaves FROM keypoints WHERE image_id == ?", (img_id,)).fetchone()[0], np.uint8)
        angles = COLMAPDatabase.blob_to_array(mnm_db.execute("SELECT angles FROM keypoints WHERE image_id == ?", (img_id,)).fetchone()[0], np.float32)
        sizes = COLMAPDatabase.blob_to_array(mnm_db.execute("SELECT sizes FROM keypoints WHERE image_id == ?", (img_id,)).fetchone()[0], np.float32)
        responses = COLMAPDatabase.blob_to_array(mnm_db.execute("SELECT responses FROM keypoints WHERE image_id == ?", (img_id,)).fetchone()[0], np.float32)
        greenIntensities = COLMAPDatabase.blob_to_array(mnm_db.execute("SELECT greenIntensities FROM keypoints WHERE image_id == ?", (img_id,)).fetchone()[0], np.uint8)
        dominantOrientations = COLMAPDatabase.blob_to_array(mnm_db.execute("SELECT dominantOrientations FROM keypoints WHERE image_id == ?", (img_id,)).fetchone()[0], np.uint8)
        matched_to_verify = COLMAPDatabase.blob_to_array(mnm_db.execute("SELECT matched FROM keypoints WHERE image_id == ?", (img_id,)).fetchone()[0], np.uint8)

        _, _, descs = get_descriptors(mnm_db, img_id)

        # Not that all are in the same order. All data from above, and xys, descs are in the same order.
        assert(img_data.xys.shape[0] == img_data.point3D_ids.shape[0] == descs.shape[0] == rows) # just for my sanity

        # read image for rgb values
        img_file = cv2.imread(os.path.join(image_path, img_data.name))

        # get data for image' keypoint here
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
            # Use int() here you fucking idiot, so it matches the rounding in create_universal_models.py
            # and you get the right green pixel value
            y = int(xy[1])
            x = int(xy[0])
            if(y >= img_file.shape[0] or x >= img_file.shape[1]):
                print(f"Skipping {img_name} because it has a keypoint outside of the image")
                continue
            brg = img_file[y, x] #opencv conventions
            blue = brg[0]
            green = brg[1]
            red = brg[2]

            octave = octaves[i]
            angle = angles[i]
            size = sizes[i]
            response = responses[i]
            greenIntensity = greenIntensities[i]
            dominantOrientation = dominantOrientations[i]
            match_to_verify = matched_to_verify[i]

            # extra sanity checks
            assert(matched == match_to_verify)
            assert(greenIntensity == green)

            # image_id INTEGER NOT NULL,
            # name TEXT NOT NULL,
            # sift BLOB NOT NULL,
            # xyz BLOB NOT NULL,
            # xy BLOB NOT NULL,
            # blue INTEGER NOT NULL,
            # green INTEGER NOT NULL,
            # red INTEGER NOT NULL,
            # octave INTEGER NOT NULL,
            # angle FLOAT NOT NULL,
            # size FLOAT NOT NULL,
            # response FLOAT NOT NULL,
            # domOrientations INTEGER NOT NULL,
            # base INTEGER NOT NULL,
            # live INTEGER NOT NULL,
            # gt INTEGER NOT NULL,
            # matched INTEGER NOT NULL

            mnm_ml_db.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                              (img_id,) + (img_name,) + (COLMAPDatabase.array_to_blob(desc),) + (COLMAPDatabase.array_to_blob(xyz),) + (
                              COLMAPDatabase.array_to_blob(xy),) + (int(blue),) + (int(green),) + (int(red),) + (int(octave),) + (float(angle),) + (float(size),) + (
                              float(response),) + (int(dominantOrientation),) + (int(is_image_base),) + (int(is_image_live),) + (int(is_image_gt),) + (int(matched),))

    mnm_ml_db.commit()

    print('Done!')

    print("Generating Data Info...")
    all_data = mnm_ml_db.execute("SELECT * FROM data ORDER BY image_id DESC").fetchall()

    all_classes = (row[-1] for row in all_data)  # binary values
    all_classes = np.array(list(all_classes))

    print(" Total Training Size: " + str(all_classes.shape[0]))
    positive_percentage = np.where(all_classes == 1)[0].shape[0] * 100 / all_classes.shape[0]
    negative_percentage = np.where(all_classes == 0)[0].shape[0] * 100 / all_classes.shape[0]
    print(" Negative Percentage %: " + str(negative_percentage))
    print(" Positive Percentage %: " + str(positive_percentage))
    return negative_percentage, positive_percentage

def create_data(params):
    print("Generating data for ML.. (using OpenCV SIFT)")
    ml_db_dir = os.path.join(params.base_path, "ML_data/")
    os.makedirs(ml_db_dir, exist_ok=True)
    # This method will read data generated from create_universal_models.py, that used OpenCV SIFT
    # No score here, you might need to add them later if you train a regressor
    # To run the method below you need to first run create_universal_models.py for the dataset you want
    negative_percentage, positive_percentage = create_all_data_opencv_sift(params)
    return negative_percentage, positive_percentage

root_path = "/media/iNicosiaData/engd_data/"
dataset = sys.argv[1] #HGE, CAB, LIN (or Other for CMU, retail shop)

if(dataset == "HGE" or dataset == "CAB" or dataset == "LIN"):
    base_path = os.path.join(root_path, "lamar", f"{dataset}_colmap_model")
    print("Base path: " + base_path)
    parameters = Parameters(base_path)
    image_path = os.path.join(root_path, "lamar/", dataset)
    negative_percentage, positive_percentage = create_data(parameters)
    np.savetxt(os.path.join(parameters.base_path, "ML_data/", f"{dataset}_classes_percentage.txt"), [negative_percentage, positive_percentage])

if(dataset == "CMU"):
    slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                    "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    for slice_name in slices_names:
        base_path = os.path.join(root_path, "cmu", f"{slice_name}", "exmaps_data")
        print("Base path: " + base_path)
        parameters = Parameters(base_path)
        image_path = os.path.join(base_path, "live", "images")
        negative_percentage, positive_percentage = create_data(parameters)
        np.savetxt(os.path.join(parameters.base_path, "ML_data/", f"CMU_{slice_name}_classes_percentage.txt"), [negative_percentage, positive_percentage])

if(dataset == "RetailShop"):
    base_path = os.path.join(root_path, "retail_shop", "slice1")
    print("Base path: " + base_path)
    parameters = Parameters(base_path)
    image_path = os.path.join(base_path, "live", "images")
    negative_percentage, positive_percentage = create_data(parameters)
    np.savetxt(os.path.join(parameters.base_path, "ML_data/", f"{dataset}_classes_percentage.txt"), [negative_percentage, positive_percentage])
