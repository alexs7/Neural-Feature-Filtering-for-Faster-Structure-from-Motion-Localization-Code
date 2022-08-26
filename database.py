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
                                                    image_id INTEGER NOT NULL,
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
            print(e)    \

    @staticmethod
    def create_db_match_no_match_data(db_file):
        sql_drop_table_if_exists = "DROP TABLE IF EXISTS data;"
        sql_create_data_table = """CREATE TABLE IF NOT EXISTS data (
                                                    image_id INTEGER NOT NULL,
                                                    sift BLOB NOT NULL,
                                                    matched INTEGER NOT NULL,
                                                    scale FLOAT NOT NULL,
                                                    orientation FLOAT NOT NULL,
                                                    x FLOAT NOT NULL,
                                                    y FLOAT NOT NULL,
                                                    greenInt INTEGER NOT NULL
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
