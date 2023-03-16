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
    def create_db_for_all_data_opencv(db_file):
        sql_drop_table_if_exists = "DROP TABLE IF EXISTS data;"
        sql_create_data_table = """CREATE TABLE IF NOT EXISTS data (
                                                image_id INTEGER NOT NULL,
                                                name TEXT NOT NULL,
                                                sift BLOB NOT NULL,
                                                xyz BLOB NOT NULL,
                                                xy BLOB NOT NULL,
                                                blue INTEGER NOT NULL,
                                                green INTEGER NOT NULL,
                                                red INTEGER NOT NULL,
                                                octave INTEGER NOT NULL,
                                                angle FLOAT NOT NULL,
                                                size FLOAT NOT NULL,
                                                response FLOAT NOT NULL,
                                                domOrientations INTEGER NOT NULL,
                                                base INTEGER NOT NULL,
                                                live INTEGER NOT NULL,
                                                gt INTEGER NOT NULL,
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
                                                blue INTEGER NOT NULL,
                                                green INTEGER NOT NULL,
                                                red INTEGER NOT NULL,
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

    def add_keypoints(self, image_id, keypoints):
        assert (len(keypoints.shape) == 2)
        assert (keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute("INSERT INTO keypoints VALUES (?, ?, ?, ?)", (image_id,) + keypoints.shape + (self.array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute("INSERT INTO descriptors VALUES (?, ?, ?, ?)", (image_id,) + descriptors.shape + (self.array_to_blob(descriptors),))

    def add_octaves_column(self):
        addColumn = "ALTER TABLE keypoints ADD COLUMN octaves BLOB"
        self.execute(addColumn)

    def add_angles_column(self):
        addColumn = "ALTER TABLE keypoints ADD COLUMN angles BLOB"
        self.execute(addColumn)

    def add_sizes_column(self):
        addColumn = "ALTER TABLE keypoints ADD COLUMN sizes BLOB"
        self.execute(addColumn)

    def add_responses_column(self):
        addColumn = "ALTER TABLE keypoints ADD COLUMN responses BLOB"
        self.execute(addColumn)

    def add_green_intensities_column(self):
        addColumn = "ALTER TABLE keypoints ADD COLUMN greenIntensities BLOB"
        self.execute(addColumn)

    def add_dominant_orientations_column(self):
        addColumn = "ALTER TABLE keypoints ADD COLUMN dominantOrientations BLOB"
        self.execute(addColumn)

    def add_matched_column(self):
        addColumn = "ALTER TABLE keypoints ADD COLUMN matched INTEGER DEFAULT 99"
        self.execute(addColumn)

    def add_images_localised_column(self):
        addColumn = "ALTER TABLE images ADD COLUMN localised INTEGER DEFAULT 0"
        self.execute(addColumn)

    def add_images_is_base_column(self):
        addColumn = "ALTER TABLE images ADD COLUMN base INTEGER DEFAULT 0"
        self.execute(addColumn)

    def add_images_is_live_column(self):
        addColumn = "ALTER TABLE images ADD COLUMN live INTEGER DEFAULT 0"
        self.execute(addColumn)

    def add_images_is_gt_column(self):
        addColumn = "ALTER TABLE images ADD COLUMN gt INTEGER DEFAULT 0"
        self.execute(addColumn)

    def replace_keypoints(self, image_id, keypoints):
        assert (len(keypoints.shape) == 2)
        assert (keypoints.shape[1] in [2, 4, 6])
        keypoints = np.asarray(keypoints, np.float32)
        self.execute("UPDATE keypoints SET rows = ? WHERE image_id = ?", (keypoints.shape[0], image_id))
        self.execute("UPDATE keypoints SET cols = ? WHERE image_id = ?", (keypoints.shape[1], image_id))
        self.execute("UPDATE keypoints SET data = ? WHERE image_id = ?", (self.array_to_blob(keypoints), image_id))

    def insert_matches(self, pair_id, rows, cols, matches):
        self.execute("INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);", (pair_id, rows, cols, self.array_to_blob(matches)))

    def replace_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute("UPDATE descriptors SET rows = ? WHERE image_id = ?", (descriptors.shape[0], image_id))
        self.execute("UPDATE descriptors SET cols = ? WHERE image_id = ?", (descriptors.shape[1], image_id))
        self.execute("UPDATE descriptors SET data = ? WHERE image_id = ?", (self.array_to_blob(descriptors), image_id))

    def update_octaves(self, image_id, octaves):
        octaves = np.asarray(octaves, np.uint8)
        self.execute("UPDATE keypoints SET octaves = ? WHERE image_id = ?", (self.array_to_blob(octaves), image_id))

    def update_angles(self, image_id, angles):
        angles = np.asarray(angles, np.float32)
        self.execute("UPDATE keypoints SET angles = ? WHERE image_id = ?", (self.array_to_blob(angles), image_id))

    def update_sizes(self, image_id, sizes):
        sizes = np.asarray(sizes, np.float32)
        self.execute("UPDATE keypoints SET sizes = ? WHERE image_id = ?", (self.array_to_blob(sizes), image_id))

    def update_responses(self, image_id, responses):
        responses = np.asarray(responses, np.float32)
        self.execute("UPDATE keypoints SET responses = ? WHERE image_id = ?", (self.array_to_blob(responses), image_id))

    def update_green_intensities(self, image_id, green_intensities):
        green_intensities = np.asarray(green_intensities, np.uint8)
        self.execute("UPDATE keypoints SET greenIntensities = ? WHERE image_id = ?", (self.array_to_blob(green_intensities), image_id))

    def update_dominant_orientations(self, image_id, dominant_orientations):
        dominant_orientations = np.asarray(dominant_orientations, np.uint8)
        self.execute("UPDATE keypoints SET dominantOrientations = ? WHERE image_id = ?", (self.array_to_blob(dominant_orientations), image_id))

    def update_matched_values(self, image_id, matched):
        matched = np.asarray(matched, np.uint8)
        self.execute("UPDATE keypoints SET matched = ? WHERE image_id = ?", (matched, image_id))

    def update_images_localised_value(self, image_id, localised):
        self.execute("UPDATE images SET localised = ? WHERE image_id = ?", (localised, image_id))

    def update_images_is_base_value(self, img_name):
        self.execute("UPDATE images SET base = ? WHERE name = ?", (1, img_name))

    def update_images_is_live_value(self, img_name):
        self.execute("UPDATE images SET live = ? WHERE name = ?", (1, img_name))

    def update_images_is_gt_value(self, img_name):
        self.execute("UPDATE images SET gt = ? WHERE name = ?", (1, img_name))

    def delete_all_matches(self):
        self.execute("DELETE FROM matches")

    def delete_all_two_view_geometries(self):
        self.execute("DELETE FROM two_view_geometries")