import sqlite3
import numpy as np
from tensorflow.keras.utils import Sequence
from database import COLMAPDatabase

class SQLiteDataGenerator(Sequence):
    def __init__(self, db_file, batch_size, input_shape, output_shape, val=False):
        self.db_file = db_file
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.val = val

        self.connection = sqlite3.connect(self.db_file, check_same_thread=False)
        self.cursor = self.connection.cursor()

        self.num_samples = self.get_num_samples()

        # If this class is called to create a validation set then the number of samples
        # is reduced to 30% of the total number of samples and the offset is set to 70%
        # The offset will still work because we are querying from the whole database (all num_samples).
        if(self.val):
            self.val_offset = int(np.ceil(self.num_samples * 0.7))
            self.num_samples = self.get_num_samples() - self.val_offset #30% of the data is used for validation
        else:
            self.val_offset = 0
            self.num_samples = int(np.ceil(self.num_samples * 0.7)) #70% of the data is used for training

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []

        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size

        # At this point the data table is permanently randomised from the train_for_nf.py script
        # If SQLiteDataGenerator was called to create training data then the offset is zero
        # and does not have any effect here. The idx will take values from 0 to (int(0.7 * num_samples) / bath_size)-1
        # If the SQLiteDataGenerator was called to create validation data then the offset is set to 70% of the total
        # number of samples and the number of samples is reduced to 30% of the total number of samples.
        # The idx will take values from 0 to (int(0.3 * num_samples) / bath_size)-1
        start_idx += self.val_offset
        end_idx += self.val_offset

        assert end_idx - start_idx == self.batch_size, "Batch size is not equal to the number of samples"

        query = f"SELECT sift, xy, blue, green, red, octave, angle, size, response, domOrientations, matched FROM data WHERE (base == 1 OR live == 1) LIMIT {start_idx}, {self.batch_size};"
        self.cursor.execute(query)
        results = self.cursor.fetchall()

        batch = np.empty([self.batch_size, 139])
        for i in range(len(results)):
            row = results[i]
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

            batch[i,0:128] = sift
            batch[i,128:130] = xy
            batch[i,130] = blue
            batch[i,131] = green
            batch[i,132] = red
            batch[i,133] = octave
            batch[i,134] = angle
            batch[i,135] = size
            batch[i,136] = response
            batch[i,137] = domOrientations
            batch[i,138] = matched

        batch_x = batch[:,0:138].astype(np.float32)
        batch_y = batch[:,138].astype(np.float32)
        return batch_x, batch_y

    def get_num_samples(self):
        query = "SELECT COUNT(*) FROM data WHERE (base == 1 OR live == 1);"
        self.cursor.execute(query)
        result = self.cursor.fetchone()
        return result[0]

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y