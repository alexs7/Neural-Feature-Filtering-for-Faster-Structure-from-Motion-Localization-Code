import numpy as np
from tqdm import tqdm
from database import COLMAPDatabase

# Methods below fetch data for TRAINING

# openCV database ha no scores so less columns
def getClassificationDataOpenCV(db_path):
    ml_db = COLMAPDatabase.connect_ML_db(db_path)

    print(" Running the select query..")
    data = ml_db.execute("SELECT sift, xy, blue, green, red, octave, angle, size, response, domOrientations, matched FROM data WHERE (base == 1 OR live == 1) ORDER BY RANDOM()").fetchall()  # use LIMIT to debug NNs

    training_data = np.empty([len(data), 139])

    for i in tqdm(range(len(data))):
        row = data[i]
        sift = COLMAPDatabase.blob_to_array(row[0], np.uint8)  # SIFT
        xy = COLMAPDatabase.blob_to_array(row[1], np.float64)  # xy image
        blue = row[2]  # blue
        green = row[3]  # green
        red = row[4]  # red
        octave = row[5]  # octave
        angle = row[6]  # angle
        size = row[7]  # size
        response = row[8]  # response
        domOrientations = row[9]  # dominant orientations
        matched = row[10]  # matched

        training_data[i, 0:128] = sift
        training_data[i, 128:130] = xy
        training_data[i, 130] = blue
        training_data[i, 131] = green
        training_data[i, 132] = red
        training_data[i, 133] = octave
        training_data[i, 134] = angle
        training_data[i, 135] = size
        training_data[i, 136] = response
        training_data[i, 137] = domOrientations
        training_data[i, 138] = matched

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