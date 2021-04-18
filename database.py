import sqlite3
from sqlite3 import Error
import sys
import numpy as np

IS_PYTHON3 = sys.version_info[0] >= 3

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    @staticmethod
    def blob_to_array(blob, dtype, shape=(-1,)):
        if IS_PYTHON3:
            return np.frombuffer(blob, dtype=dtype).reshape(*shape)
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

    @staticmethod
    def create_db_for_training_data(db_file):
        sql_drop_table_if_exists = "DROP TABLE IF EXISTS data;"
        sql_create_data_table = """CREATE TABLE IF NOT EXISTS data (
                                                sift BLOB NOT NULL,
                                                score FLOAT NOT NULL,
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
    def create_db_for_test_data(db_file):
        sql_drop_table_if_exists = "DROP TABLE IF EXISTS data;"
        sql_create_data_table = """CREATE TABLE IF NOT EXISTS data (
                                                sift BLOB NOT NULL,
                                                score FLOAT NOT NULL,
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

    # xyz -> np.float64, xy -> np.float64, score,pred_score -> as it is (I think you may be able to use np.float64/32)
    # sift -> np.uint8
    @staticmethod
    def create_db_for_all_data(db_file):
        sql_drop_table_if_exists = "DROP TABLE IF EXISTS data;"
        sql_create_data_table = """CREATE TABLE IF NOT EXISTS data (
                                                image_id INTEGER NOT NULL,
                                                name TEXT NOT NULL,
                                                sift BLOB NOT NULL,
                                                score FLOAT NOT NULL,
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
