import numpy as np
from tqdm import tqdm
from database import COLMAPDatabase

# Methods below fetch data for TRAINING

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