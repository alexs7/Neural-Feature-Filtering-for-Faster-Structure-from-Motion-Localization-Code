from sklearn import preprocessing
from database import COLMAPDatabase
import numpy as np

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
    ml_db = COLMAPDatabase.connect_ML_db(db_path)

    print(" Running the select query..")
    data = ml_db.execute("SELECT sift, matched FROM data").fetchall()  # guarantees same order - maybe ?

    sift_vecs = (COLMAPDatabase.blob_to_array(row[0], np.uint8) for row in data)
    sift_vecs = np.array(list(sift_vecs))

    classes = (row[1] for row in data)  # binary values
    classes = np.array(list(classes))

    shuffled_idxs = np.random.permutation(sift_vecs.shape[0])
    sift_vecs = sift_vecs[shuffled_idxs]
    classes = classes[shuffled_idxs]

    print("Total Training Size: " + str(sift_vecs.shape[0]))

    return sift_vecs, classes

# returns data for regression and classification
def getCombinedData(db_path, score_name):
    # score_name is either score_per_image, score_per_session, score_visibility
    ml_db = COLMAPDatabase.connect_ML_db(db_path)

    print(" Running the select query..")
    data = ml_db.execute("SELECT sift, "+score_name+", matched FROM data").fetchall()  # guarantees same order - maybe ?

    sift_vecs = (COLMAPDatabase.blob_to_array(row[0], np.uint8) for row in data)
    sift_vecs = np.array(list(sift_vecs))

    scores = (row[1] for row in data)  # continuous values
    scores = np.array(list(scores))

    classes = (row[2] for row in data)  # binary values
    classes = np.array(list(classes))

    shuffled_idxs = np.random.permutation(sift_vecs.shape[0])
    sift_vecs = sift_vecs[shuffled_idxs]
    scores = scores[shuffled_idxs]
    classes = classes[shuffled_idxs]

    print("Total Training Size: " + str(sift_vecs.shape[0]))

    return sift_vecs, scores, classes