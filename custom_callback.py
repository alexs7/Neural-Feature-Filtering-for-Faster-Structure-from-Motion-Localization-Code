import tensorflow as tf
from tensorflow import keras
from database import COLMAPDatabase
from parameters import Parameters
import numpy as np
from query_image import read_images_binary, load_images_from_text_file, get_localised_image_by_names, get_query_images_pose_from_images, get_intrinsics_from_camera_bin
from point3D_loader import read_points3d_default, get_points3D_xyz
from feature_matching_generator_ML import feature_matcher_wrapper
from ransac_prosac import ransac, ransac_dist, prosac
from get_scale import calc_scale_COLMAP_ARCORE
from benchmark import benchmark

# Before this you have to run "python3 prepare_comparison_data.py" to prepare and save comparison data
class CustomCallback(keras.callbacks.Callback):

    def __init__(self, cust_log_dir):
        self.cust_log_dir = cust_log_dir
        self.writer = tf.summary.create_file_writer(self.cust_log_dir)

    # def on_train_begin(self, logs=None):
    #     # load comparison data here
    #     keys = list(logs.keys())
    #     print("Stop training; got log keys: {}".format(keys))

    # def on_train_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop training; got log keys: {}".format(keys))
    #
    # def on_epoch_begin(self, epoch, logs=None):
    #     keys = list(logs.keys())
    #     print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    # def on_epoch_end(self, epoch, logs=None):
    #
    #     import pdb
    #     pdb.set_trace()

        # (this is pretty much the evaluation that will be used after teh model has trained or after each epoch)
        # 1 - get the gt poses
        # 2 - predict matchable features using current model (use those only for matching)
        # 3 - use all features for matching
        # 4 - use random features for matching
        # 5 - get inliers, ouliers, iterations, time (including prediction time), t_error, r_error mean for all epoch_images (or test - not actually test though)
        # 6 - compare the errors from 2,3 and 4 against the gt poses from COLMAP
        # 7 - pass the numbers to tensorboard

        # This is what writes to the file, after training is done
        # with self.writer.as_default():
        #     val = tf.summary.scalar("my_metric", 0.5, step=step)
        #
        # self.writer.flush()

        # Extra code
        # import pdb  # pdb.set_trace()
    # writer = tf.summary.create_file_writer(self.tb_callback.log_dir)
    # #self.tb_callback.writer
    # writer = tf.summary.create_file_writer(self.tb_callback.log_dir)
    # writer = tf.summary.create_file_writer("/tmp/mylogs/eager")
        # tf.summary.scalar(tag, value, step=step)
        #
        # for name, value in items_to_write.items():
        #     summary = tf.summary.Summary()
        #     summary_value = summary.value.add()
        #     summary_value.simple_value = value
        #     summary_value.tag = name
        #     writer.add_summary(summary, self.step_number)
        #     writer.flush()
        #
        # self.step_number += 1

        # with self.file_writer.as_default():
        #     tf.summary.scalar('error_test', data=epoch, step=epoch)
        #     import pdb
        #     pdb.set_trace()
        #     self.file_writer.flush() # this might not be needed if "flush_millis = 2000" is set

        # keys = list(logs.keys())
        # print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    # def on_test_begin(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Start testing; got log keys: {}".format(keys))
    #
    # def on_test_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop testing; got log keys: {}".format(keys))
    #
    # def on_predict_begin(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Start predicting; got log keys: {}".format(keys))
    #
    # def on_predict_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop predicting; got log keys: {}".format(keys))
    #
    # def on_train_batch_begin(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Training: start of batch {}; got log keys: {}".format(batch, keys))
    #
    # def on_train_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
    #
    # def on_test_batch_begin(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))
    #
    # def on_test_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))
    #
    # def on_predict_batch_begin(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))
    #
    # def on_predict_batch_end(self, batch, logs=None):
    #     keys = list(logs.keys())
    #     print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))