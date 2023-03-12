# This file returns the direct data to be used for model for training and testing
# Separated in 2 sections training and testing
import os.path
import cv2
import numpy as np
from tqdm import tqdm
from database import COLMAPDatabase
from point3D_loader import read_points3d_default
from query_image import read_images_binary, get_image_decs, get_localised_image_by_names

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

# openCV database ha no scores so less columns
def getClassificationDataOpenCV(db_path):
    # db fields:
    # table : data
    # image_id INTEGER NOT NULL,
    # name TEXT NOT NULL,
    # sift BLOB NOT NULL,
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
        training_data[i,128:130] = COLMAPDatabase.blob_to_array(row[4], np.float64) #xy image
        training_data[i,130:131] = row[5] #blue
        training_data[i,131:132] = row[6] #green
        training_data[i,132:133] = row[7] #red
        training_data[i,133:134] = row[8] #matched

    print("Total Training Size: " + str(training_data.shape[0]))
    return training_data

def getClassificationDataPM(db_path):
    ml_db = COLMAPDatabase.connect_ML_db(db_path)

    print(" Running the select query..")
    data = ml_db.execute("SELECT sift, matched FROM data ORDER BY RANDOM()").fetchall()

    sift_vecs = (COLMAPDatabase.blob_to_array(row[0], np.uint8) for row in data)
    sift_vecs = np.array(list(sift_vecs))

    classes = (row[1] for row in data)  # binary values
    classes = np.array(list(classes))

    print("Total Training Size: " + str(sift_vecs.shape[0]))

    return sift_vecs, classes

# Methods below fetch data used for TESTING, i.e. comparing to GT data

def get_MnM_data(parameters):
    # Not returning SIFT here are MnM paper does not use SIFT
    gt_db_mnm = COLMAPDatabase.connect(parameters.gt_db_path_mnm)
    gt_image_names = np.loadtxt(parameters.query_gt_images_txt_path_mnm, dtype=str)  # only gt images
    localised_qt_images_names = get_localised_image_by_names(gt_image_names, parameters.gt_model_images_path_mnm)  # only gt images (localised only)
    localised_gt_images_ids = gt_db_mnm.execute("SELECT image_id FROM images WHERE name IN ({})".format(",".join(["?"] * len(localised_qt_images_names))), localised_qt_images_names).fetchall()
    localised_gt_images_ids = [x[0] for x in localised_gt_images_ids]
    all_kps_data = gt_db_mnm.execute("SELECT rows, cols, data, octaves, angles, sizes, responses, greenIntensities, dominantOrientations, matched FROM keypoints WHERE image_id IN ({})".format(",".join(["?"] * len(localised_qt_images_names))), localised_gt_images_ids).fetchall()

    # from C++ code, order of features is:
    # features.at < float > (j, 0) = keypoints[j].pt.x;
    # features.at < float > (j, 1) = keypoints[j].pt.y;
    # features.at < float > (j, 2) = keypoints[j].octave;
    # features.at < float > (j, 3) = keypoints[j].angle;
    # features.at < float > (j, 4) = keypoints[j].size;
    # features.at < float > (j, 5) = keypoints[j].response;
    # features.at < float > (j, 6) = Green.at < double > (j, 0);
    # features.at < float > (j, 7) = domOrientations.at < double > (j, 0);

    data = np.empty([0,9])
    images_to_examine = 0
    for row in tqdm(all_kps_data): #each row is a row of data for that image
        rows_no = row[0]
        cols_no = row[1]
        kps_xy = COLMAPDatabase.blob_to_array(row[2], np.float32).reshape(rows_no, cols_no) #(x,y) shape (rows_no, 2)
        octaves = COLMAPDatabase.blob_to_array(row[3], np.uint8).reshape(rows_no, 1) #octaves (rows_no, 1)
        angles = COLMAPDatabase.blob_to_array(row[4], np.float32).reshape(rows_no, 1)
        sizes = COLMAPDatabase.blob_to_array(row[5], np.float32).reshape(rows_no, 1)
        responses = COLMAPDatabase.blob_to_array(row[6], np.float32).reshape(rows_no, 1)
        greenIntensities = COLMAPDatabase.blob_to_array(row[7], np.uint8).reshape(rows_no, 1)
        dominantOrientations = COLMAPDatabase.blob_to_array(row[8], np.uint8).reshape(rows_no, 1)
        if(row[9] == 99 or COLMAPDatabase.blob_to_array(row[9], np.uint8).shape[0] == 0):
            # At this point for various reasons I did not add matched data to the database
            # for this specific localised image, so I will just skip it
            # Check create_universal_models.py for more info
            # The second case happens when an image has keypoints but no image.xys for some reason (because of COLMAP most probably).
            continue
        images_to_examine += 1
        matched = COLMAPDatabase.blob_to_array(row[9], np.uint8).reshape(rows_no, 1)
        image_data = np.c_[kps_xy, octaves, angles, sizes, responses, greenIntensities, dominantOrientations, matched]
        data = np.r_[data, image_data]
    print(f"Total gt images to examine: {images_to_examine} / {len(localised_gt_images_ids)}")
    return data

def get_default_data(parameters, image_path):
    # return SIFT here (COLMAP's) as PM and my method NF, use SIFT
    gt_db = COLMAPDatabase.connect(parameters.gt_db_path)
    gt_model_images = read_images_binary(parameters.gt_model_images_path)
    gt_points_3D = read_points3d_default(parameters.gt_model_points3D_path)

    gt_image_names = np.loadtxt(parameters.gt_query_images_path, dtype=str)  # only gt images
    localised_qt_images_names = get_localised_image_by_names(gt_image_names, parameters.gt_model_images_path)  # only gt images (localised only)
    localised_gt_images_ids = gt_db.execute("SELECT image_id FROM images WHERE name IN ({})".format(",".join(["?"] * len(localised_qt_images_names))), localised_qt_images_names).fetchall()
    localised_gt_images_ids = [x[0] for x in localised_gt_images_ids]
    print(f"Total number of gt localised images: {len(localised_qt_images_names)}")
    total_keypoints_no = gt_db.execute("SELECT SUM(rows) FROM keypoints WHERE image_id IN ({})".format(",".join(["?"] * len(localised_gt_images_ids))), localised_gt_images_ids).fetchone()[0]

    # allocate array to store all test / gt data
    # SIFT + XY + RGB = 134
    gt_data = np.empty([total_keypoints_no, 134])
    k = 0
    for img_id in tqdm(localised_gt_images_ids):  # only loop through gt images that were localised from query_name.txt
        descs = get_image_decs(gt_db, img_id)
        img_data = gt_model_images[int(img_id)]
        # sanity checks
        assert (img_data.id == img_id)
        assert (img_data.xys.shape[0] == img_data.point3D_ids.shape[0] == descs.shape[0])

        img_file = cv2.imread(os.path.join(image_path, img_data.name)) #at this point I am looking at gt images only
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
            # OpenCV uses BRG, not RGB
            blue = brg[0]
            green = brg[1]
            red = brg[2]

            # need to store in the same order as in the training data found in, getClassificationData()
            # SIFT + XY + RGB + Matched
            gt_data[k, :] = np.append(desc, [xy[0], xy[1], blue, green, red, matched])
            k += 1
    return gt_data
