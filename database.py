import sqlite3
from sqlite3 import Error
import sys
import numpy as np

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    @staticmethod
    def blob_to_array(blob, dtype, shape=(-1,)):
        if IS_PYTHON3:
            return np.fromstring(blob, dtype=dtype).reshape(*shape)
        else:
            return np.frombuffer(blob, dtype=dtype).reshape(*shape)

    @staticmethod
    def array_to_blob(array):
        if IS_PYTHON3:
            return array.tostring()
        else:
            return np.getbuffer(array)

    @staticmethod
    def connect_ML_db(path):
        conn = sqlite3.connect(path)
        return conn

    # xyz -> np.float64, xy -> np.float64, score,pred_score -> as it is (I think you may be able to use np.float64/32)
    # sift -> np.uint8
    @staticmethod
    def create_db_for_all_data(db_file):
        sql_drop_table_if_exists = "DROP TABLE IF EXISTS data;"
        sql_create_data_table = """CREATE TABLE IF NOT EXISTS data (
                                                image_id INTEGER NOT NULL,
                                                name TEXT NOT NULL,
                                                sift BLOB NOT NULL,
                                                score_per_image FLOAT NOT NULL,
                                                score_per_session FLOAT NOT NULL,
                                                score_visibility FLOAT NOT NULL,
                                                xyz BLOB NOT NULL,
                                                xy BLOB NOT NULL,
                                                matched INTEGER NOT NULL
                                            );"""
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            conn.execute(sql_drop_table_if_exists)
            conn.commit()
            conn.execute(sql_create_data_table)
            conn.commit()
            return conn
        except Error as e:
            print(e)

    @staticmethod
    def create_db_predicting_matchability_data(db_file):
        sql_drop_table_if_exists = "DROP TABLE IF EXISTS data;"
        sql_create_data_table = """CREATE TABLE IF NOT EXISTS data (
                                                    sift BLOB NOT NULL,
                                                    matched INTEGER NOT NULL
                                                );"""
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            conn.execute(sql_drop_table_if_exists)
            conn.commit()
            conn.execute(sql_create_data_table)
            conn.commit()
            return conn
        except Error as e:
            print(e)

    @staticmethod
    def create_db_match_no_match_data(db_file):
        # x, y, octave, angle, size, response
        sql_drop_table_if_exists = "DROP TABLE IF EXISTS data;"
        sql_create_data_table = """CREATE TABLE IF NOT EXISTS data (
                                                    x FLOAT NOT NULL,
                                                    y FLOAT NOT NULL,
                                                    octave FLOAT NOT NULL,
                                                    angle FLOAT NOT NULL,
                                                    size FLOAT NOT NULL,
                                                    response FLOAT NOT NULL,
                                                    domOrientations FLOAT NOT NULL,
                                                    green FLOAT NOT NULL,
                                                    matched INTEGER NOT NULL,
                                                    imageId INTEGER NOT NULL
                                                    );"""
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            conn.execute(sql_drop_table_if_exists)
            conn.commit()
            conn.execute(sql_create_data_table)
            conn.commit()
            return conn
        except Error as e:
            print(e)

    @staticmethod
    def create_db_for_visual_data(db_file):
        sql_drop_table_if_exists = "DROP TABLE IF EXISTS data;"
        sql_create_data_table = """CREATE TABLE IF NOT EXISTS data (
                                                sift BLOB NOT NULL,
                                                pred_score FLOAT NOT NULL,
                                                score FLOAT NOT NULL, 
                                                xyz BLOB NOT NULL
                                            );"""
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            conn.execute(sql_drop_table_if_exists)
            conn.commit()
            conn.execute(sql_create_data_table)
            conn.commit()
            return conn
        except Error as e:
            print(e)

    def add_keypoints(self, image_id, keypoints):
        assert (len(keypoints.shape) == 2)
        assert (keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute("INSERT INTO keypoints VALUES (?, ?, ?, ?)", (image_id,) + keypoints.shape + (self.array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute("INSERT INTO descriptors VALUES (?, ?, ?, ?)", (image_id,) + descriptors.shape + (self.array_to_blob(descriptors),))

    def dominant_orientations_column_exists(self):
        cols = self.execute("PRAGMA table_info('keypoints');").fetchall()
        for col in cols:
            if(col[1] == 'dominantOrientations'):
                return True
        return False

    def add_dominant_orientations_column(self):
        addColumn = "ALTER TABLE keypoints ADD COLUMN dominantOrientations BLOB"
        self.execute(addColumn)

    def replace_keypoints(self, image_id, keypoints, dominant_orientations):
        assert (len(keypoints.shape) == 2)
        assert (keypoints.shape[1] in [2, 4, 6])
        assert dominant_orientations.shape[0] == len(keypoints)

        # delete first
        self.execute("DELETE FROM keypoints WHERE image_id = " + "'" + str(image_id) + "'")
        # insert again
        keypoints = np.asarray(keypoints, np.float32)
        dominant_orientations = np.asarray(dominant_orientations, np.uint8) #np.uint8 is OK the array contains just int increment values
        self.execute("INSERT INTO keypoints VALUES (?, ?, ?, ?, ?)", (image_id,) + keypoints.shape + (self.array_to_blob(keypoints),) + (self.array_to_blob(dominant_orientations),))

    def replace_descriptors(self, image_id, descriptors):
        # delete first
        self.execute("DELETE FROM descriptors WHERE image_id = " + "'" + str(image_id) + "'")
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute("INSERT INTO descriptors VALUES (?, ?, ?, ?)", (image_id,) + descriptors.shape + (self.array_to_blob(descriptors),))

    def delete_all_matches(self):
        self.execute("DELETE FROM matches")

    def delete_all_two_view_geometries(self):
        self.execute("DELETE FROM two_view_geometries")