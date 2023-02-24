# This file returns the direct data to be used for model for training and testing
# Separated in 2 sections training and testing
import os.path
import cv2
from tqdm import tqdm
from database import COLMAPDatabase
import numpy as np
from point3D_loader import read_points3d_default
from query_image import load_images_from_text_file, read_images_binary, get_image_id, get_image_decs, get_descriptors, get_image_data, get_localised_image_by_names, \
    get_total_number_of_valid_keypoints


# Methods below fetch data for TRAINING

def getRegressionData(db_path, score_name, train_on_matched_only = True):
    # score_name is either score_per_image, score_per_session, score_visibility
    ml_db = COLMAPDatabase.connect_ML_db(db_path)

    if(train_on_matched_only):
        print("Fetching only matched features..")
        data = ml_db.execute("SELECT sift, "+score_name+" FROM data WHERE matched = 1").fetchall()
    else:
        print("Fetching all features..")
        data = ml_db.execute("SELECT sift, "+score_name+" FROM data").fetchall()

    sift_vecs = (COLMAPDatabase.blob_to_array(row[0], np.uint8) for row in data)
    sift_vecs = np.array(list(sift_vecs))

    scores = (row[1] for row in data)  # continuous values
    scores = np.array(list(scores))

    shuffled_idxs = np.random.permutation(sift_vecs.shape[0])
    sift_vecs = sift_vecs[shuffled_idxs]
    scores = scores[shuffled_idxs]

    print("Total Training Size: " + str(sift_vecs.shape[0]))

    return sift_vecs, scores

def getClassificationData(db_path):
    # db fields:
    # table : data
    # image_id INTEGER NOT NULL,
    # name TEXT NOT NULL,
    # sift BLOB NOT NULL,
    # score_per_image FLOAT NOT NULL,
    # score_per_session FLOAT NOT NULL,
    # score_visibility FLOAT NOT NULL,
    # xyz BLOB NOT NULL,
    # xy BLOB NOT NULL,
    # blue INTEGER NOT NULL,
    # green INTEGER NOT NULL,
    # red INTEGER NOT NULL,
    # matched INTEGER NOT NULL
    ml_db = COLMAPDatabase.connect_ML_db(db_path)

    print(" Running the select query..")
    data = ml_db.execute("SELECT * FROM data ORDER BY RANDOM()").fetchall()  # use LIMIT to debug NNs

    training_data = np.empty([len(data), 134])

    for i in tqdm(range(len(data))):
        row = data[i]
        training_data[i,0:128] = COLMAPDatabase.blob_to_array(row[2], np.uint8) #SIFT
        training_data[i,128:130] = COLMAPDatabase.blob_to_array(row[7], np.float64) #xy image
        training_data[i,130:131] = row[8] #blue
        training_data[i,131:132] = row[9] #green
        training_data[i,132:133] = row[10] #red
        training_data[i,133:134] = row[11] #matched

    print("Total Training Size: " + str(training_data.shape[0]))
    return training_data

def getClassificationDataPM(db_path):
    ml_db = COLMAPDatabase.connect_ML_db(db_path)

    print(" Running the select query..")
    data = ml_db.execute("SELECT sift, matched FROM data ORDER BY RANDOM()").fetchall()  # guarantees same order - maybe ?

    sift_vecs = (COLMAPDatabase.blob_to_array(row[0], np.uint8) for row in data)
    sift_vecs = np.array(list(sift_vecs))

    classes = (row[1] for row in data)  # binary values
    classes = np.array(list(classes))

    print("Total Training Size: " + str(sift_vecs.shape[0]))

    return sift_vecs, classes

# Methods below fetch ground truth data for TESTING, binary classification models

# Note that this will return less data than the PM/NN equivalent as less query_name.txt are localised (OpenCV localised less images)
def getClassificationDataTestingMnM(mnm_base_path, image_path):
    db_gt_mnm_path = os.path.join(mnm_base_path, "gt/database.db") #openCV db + extra MnM data
    db_gt_mnm = COLMAPDatabase.connect(db_gt_mnm_path)  # remember this database holds the OpenCV descriptors
    query_images_bin_path_mnm = os.path.join(mnm_base_path, "gt/output_opencv_sift_model/images.bin")
    gt_model_images_mnm = read_images_binary(query_images_bin_path_mnm)

    query_gt_images_txt_path = os.path.join(mnm_base_path, "gt/query_name.txt") #this file was copied over from Exmaps
    gt_image_names = np.loadtxt(query_gt_images_txt_path, dtype=str)
    gt_points_3D_mnm = read_points3d_default(os.path.join(mnm_base_path, "gt/output_opencv_sift_model/points3D.bin"))

    print("Reading gt images .bin (localised MnM)...")
    # This is loading the images.bin file again so it might be a bit inefficient to do it twice (I do it above too).
    localised_query_images_names = get_localised_image_by_names(gt_image_names, query_images_bin_path_mnm)  # only gt images (localised only)

    print(f"Total number of gt localised images: {len(localised_query_images_names)}")

    # The method below returns the total number of data points (keypoints/ descs whatever) for all gt images
    # A data point is a SIFT + XY + RGB + ... + matched for a pixel in a gt image
    # The reason I am calling this is to preallocate an array to store all the data
    # Otherwise I would have to use a list and append to it, which is very inefficient
    # The code in the method is very similar to the code below.
    total_number_keypoints = get_total_number_of_valid_keypoints(localised_query_images_names, db_gt_mnm, gt_model_images_mnm, image_path, gt_points_3D_mnm)

    gt_data = np.empty([total_number_keypoints, 9])
    k=0
    for image_name in tqdm(localised_query_images_names): #only loop through gt images that were localised from query_name.txt
        img_id = int(get_image_id(db_gt_mnm, image_name))

        image_gt_path = os.path.join(image_path, image_name)
        qt_image_file = cv2.imread(image_gt_path)  # no need for cv2.COLOR_BGR2RGB here as G is in the middle anw
        test_data_img = get_image_data(db_gt_mnm, gt_points_3D_mnm, gt_model_images_mnm, img_id, qt_image_file)

        for i in range(len(test_data_img)): #as many keypoints as detected
            sample = test_data_img[i, :]
            x = sample[0]
            y = sample[1]
            octave = sample[2]
            angle = sample[3]
            size = sample[4]
            response = sample[5]
            green_intensity = sample[6]
            dominantOrientation = sample[7]
            matched = sample[8] #can use astype(np.int64) here
            gt_data[k,:] = [x,y,octave,angle,size,response,green_intensity,dominantOrientation,matched]
            k += 1
    assert k == total_number_keypoints
    return gt_data

# This can work for both PM 2014 and NF (mine)
def getClassificationDataTesting(base_path, image_path):
    db_gt_path = os.path.join(base_path, "gt/database.db")
    db_gt = COLMAPDatabase.connect(db_gt_path)
    query_images_bin_path = os.path.join(base_path, "gt/model/images.bin")
    gt_model_images = read_images_binary(query_images_bin_path)

    query_gt_images_txt_path = os.path.join(base_path, "gt/query_name.txt") #only gt_images names
    gt_image_names = np.loadtxt(query_gt_images_txt_path, dtype=str)
    gt_points_3D = read_points3d_default(os.path.join(base_path, "gt/model/points3D.bin"))

    print("Reading gt images .bin (localised)...")
    localised_query_images_names = get_localised_image_by_names(gt_image_names, query_images_bin_path)  # only gt images (localised only)

    print(f"Total number of gt localised images: {len(localised_query_images_names)}")

    # The method below returns the total number of data points (keypoints/ descs whatever) for all gt images
    # A data point is a SIFT + XY + RGB + ... + matched for a pixel in a gt image
    # The reason I am calling this is to preallocate an array to store all the data
    # Otherwise I would have to use a list and append to it, which is very inefficient
    # The code in the method is very similar to the code below.
    total_number_keypoints = get_total_number_of_valid_keypoints(localised_query_images_names, db_gt, gt_model_images, image_path, gt_points_3D)

    # allocate array to store all test / gt data
    gt_data = np.empty([total_number_keypoints, 134])
    k=0
    for image_name in tqdm(localised_query_images_names): #only loop through gt images that were localised from query_name.txt
        img_id = get_image_id(db_gt, image_name)
        img_data = gt_model_images[int(img_id)]
        assert (img_data.name == image_name)

        descs = get_image_decs(db_gt, img_id)
        assert(img_data.xys.shape[0] == img_data.point3D_ids.shape[0] == descs.shape[0]) # just for my sanity

        img_file = cv2.imread(os.path.join(image_path, img_data.name))
        for i in range(img_data.point3D_ids.shape[0]):  # can loop through descs or img_data.xys - same thing
            current_point3D_id = img_data.point3D_ids[i]

            if (current_point3D_id == -1):  # means feature (or keypoint) is unmatched
                matched = 0
            else:
                assert i in gt_points_3D[current_point3D_id].point2D_idxs
                matched = 1

            desc = descs[i]  # np.uint8
            xy = img_data.xys[i]  # np.float64, same as xyz
            y = np.round(xy[1]).astype(int)
            x = np.round(xy[0]).astype(int)
            if (y >= img_file.shape[0] or x >= img_file.shape[1]):
                continue
            brg = img_file[y, x]  # opencv conventions
            blue = brg[0]
            green = brg[1]
            red = brg[2]

            # need to store in the same order as in the training data found in, getClassificationData()
            # SIFT + XY + RGB + Matched
            gt_data[k, :] = np.append(desc,[xy[0], xy[1], blue, green, red, matched])
            k += 1
    assert k == total_number_keypoints
    return gt_data
