# This file returns the direct data to be used for model for training and testing
# Separated in 2 sections training and testing
import os.path
import cv2
import numpy as np
from tqdm import tqdm
from database import COLMAPDatabase
from point3D_loader import read_points3d_default
from query_image import read_images_binary, get_image_id, get_image_decs, get_image_data_mnm, get_localised_image_by_names, get_queryDescriptors, \
    countDominantOrientations

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

def getClassificationDataTestingMnM(image_names, db_path, images_path, points3D, model_images):
    db_gt_mnm = COLMAPDatabase.connect(db_path)
    total_number_keypoints = db_gt_mnm.execute("SELECT SUM(rows) FROM keypoints").fetchone()[0]
    # gt_data is the concatenated data of all images that will be used to generate binary classification model stats
    # gt_image_data is the per image data that will be used to generate the image pose data
    gt_data = np.empty([total_number_keypoints, 9])
    k = 0
    images_data = {}
    for image_name in tqdm(image_names):
        img_id = get_image_id(db_gt_mnm, image_name)
        model_image = model_images[int(img_id)]
        assert (model_image.name == image_name)
        qt_image_file = cv2.imread(os.path.join(images_path, image_name))  # no need for cv2.COLOR_BGR2RGB here as G is in the middle anw
        kp_db_row = db_gt_mnm.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = " + "'" + str(img_id) + "'").fetchone()
        cols = kp_db_row[1]
        rows = kp_db_row[0]
        kps = COLMAPDatabase.blob_to_array(kp_db_row[2], np.float32)
        kps = kps.reshape([rows, cols])
        descs = get_queryDescriptors(db_gt_mnm, img_id)

        sift = cv2.SIFT_create()  # just to verify that the descriptors number are the same from format_data_for_match_no_match.py
        opencv_kps, opencv_descs = sift.detectAndCompute(qt_image_file, None)

        # For some fucking reason the keypoints are not saved the order they are extracted in the db in
        # format_data_for_match_no_match.py file. So I need to sort the DB data by x coordinate as done in OpenCV.
        sift_data_db = np.c_[kps, descs]
        sift_data_db_sorted = sift_data_db[sift_data_db[:, 0].argsort()]
        dominantOrientations = countDominantOrientations(sift_data_db_sorted[:,0:2])

        # now the model's xys need to be sorted too along with the point3D_ids
        model_xys_point3D_id = np.c_[model_image.xys, model_image.point3D_ids]
        model_xys_point3D_id_sorted = model_xys_point3D_id[model_xys_point3D_id[:, 0].argsort()]

        # at this point the db data and the model data, and opencv are sorted by x coordinate

        # extreme sanity check
        assert (model_image.xys.shape[0] == model_image.point3D_ids.shape[0] == rows == descs.shape[0] == len(opencv_descs) == len(dominantOrientations) == len(kps))

        # Now I need the match values and green values for each keypoint
        matched_values = []  # for each keypoint (x,y)/desc same thing
        green_intensities = []  # for each keypoint (x,y)/desc same thing

        for i in range(model_xys_point3D_id_sorted.shape[0]):
            current_point3D_id = model_xys_point3D_id_sorted[i, 2]
            x = model_xys_point3D_id_sorted[i, 0]
            y = model_xys_point3D_id_sorted[i, 1]

            if (current_point3D_id == -1):  # means feature is unmatched
                matched = 0
                green_intensity = qt_image_file[int(y), int(x)][1]  # reverse indexing
            else:
                # this is to make sure the current image is in the point3D's image_ids
                # i.e the image is looking at the point3D
                assert int(img_id) in points3D[current_point3D_id].image_ids
                matched = 1
                green_intensity = qt_image_file[int(y), int(x)][1]  # reverse indexing, BGR
            matched_values.append(matched)
            green_intensities.append(green_intensity)

        matched_values = np.array(matched_values).reshape(rows, 1)
        green_intensities = np.array(green_intensities).reshape(rows, 1)

        # from C++ code, order of features is:
        # features.at < float > (j, 0) = keypoints[j].pt.x;
        # features.at < float > (j, 1) = keypoints[j].pt.y;
        # features.at < float > (j, 2) = keypoints[j].octave;
        # features.at < float > (j, 3) = keypoints[j].angle;
        # features.at < float > (j, 4) = keypoints[j].size;
        # features.at < float > (j, 5) = keypoints[j].response;
        # features.at < float > (j, 6) = Green.at < double > (j, 0);
        # features.at < float > (j, 7) = domOrientations.at < double > (j, 0);

        # they all will have the same order as the above variables
        octaves = [kp.octave for kp in opencv_kps]
        angles = [kp.angle for kp in opencv_kps]
        sizes = [kp.size for kp in opencv_kps]
        responses = [kp.response for kp in opencv_kps]

        image_data = np.c_[sift_data_db_sorted[:,0:2], octaves, angles, sizes, responses, green_intensities, dominantOrientations, matched_values]
        images_data[image_name] = image_data

        for i in range(len(image_data)): #as many keypoints as detected
            sample = image_data[i, :]
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
    return gt_data, images_data

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
