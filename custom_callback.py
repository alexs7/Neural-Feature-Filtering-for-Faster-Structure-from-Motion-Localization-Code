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

class CustomCallback(keras.callbacks.Callback):

    def __init__(self, cust_log_dir):
        self.cust_log_dir = cust_log_dir
        self.writer = tf.summary.create_file_writer(self.cust_log_dir)

    def on_train_begin(self, logs=None):
        print("Setting up...")
        # the "gt" here means "after_epoch_data" pretty much
        db_gt_path = "colmap_data/Coop_data/slice1/ML_data/after_epoch_data/after_epoch_database.db"
        query_images_bin_path = "colmap_data/Coop_data/slice1/ML_data/after_epoch_data/new_model/images.bin"
        query_images_path = "colmap_data/Coop_data/slice1/ML_data/after_epoch_data/images_list.txt"
        query_cameras_bin_path = "colmap_data/Coop_data/slice1/ML_data/after_epoch_data/new_model/cameras.bin"
        live_points3D_bin_path = "colmap_data/Coop_data/slice1/ML_data/after_epoch_data/original_live_data/model/points3D.bin"

        db_gt = COLMAPDatabase.connect(db_gt_path) # you need this database to get the query images descs as they do not exist in the live db!
        query_images = read_images_binary(query_images_bin_path)
        query_images_names = load_images_from_text_file(query_images_path)
        localised_query_images_names = get_localised_image_by_names(query_images_names, query_images_bin_path)
        query_images_ground_truth_poses = get_query_images_pose_from_images(localised_query_images_names, query_images)

        # Againg these two because points3D will have different order.
        # epoch query points
        # epoch_points3D_bin_path = "colmap_data/Coop_data/slice1/ML_data/after_epoch_data/new_model/points3D.bin"
        # points3D_epoch = read_points3d_default(epoch_points3D_bin_path)
        # points3D_xyz_epoch = get_points3D_xyz(points3D_epoch)

        # live points
        # Note: you will need to run this first, "get_points_3D_mean_desc_single_model.py" - to get the 3D points avg descs from the live model.
        # use the folder original_live_data/ otherwise you will be using the epoch image/3D point descriptors if you use the new_model
        # also you will need the scale between the colmap poses and the ARCore poses (for 2020-06-22 the 392 images are from morning run)
        # Matching will happen from the query images (epoch images) on the live model, otherwise if you use the epoch model it will be "cheating"
        # as the descriptors from the epoch images that you are trying to match will already be in the epoch model. Just use the epoch model for ground truth pose errors comparisons.
        train_descriptors_live = np.load('colmap_data/Coop_data/slice1/ML_data/after_epoch_data/original_live_data/avg_descs.npy').astype(np.float32)
        points3D_live = read_points3d_default(live_points3D_bin_path)
        points3D_xyz_live = get_points3D_xyz(points3D_live)

        K = get_intrinsics_from_camera_bin(query_cameras_bin_path, 3) #3 because 1 -base, 2 -live, 3 -epoch images
        # for ar_core data
        ar_core_poses_path = 'colmap_data/Coop_data/slice1/ML_data/after_epoch_data/test_images/2020-06-22/arcore_poses/'
        colmap_poses_path = query_images_bin_path #just for clarity purposes
        scale = calc_scale_COLMAP_ARCORE(ar_core_poses_path, colmap_poses_path)
        print("Scale: " + str(scale))

        print("Feature matching random and vanillia descs..")
        # db_gt, again because we need the descs from the query images
        ratio_test_val = 0.75
        # random 80 ones - why 80 ?
        random_no = 80
        random_matches = feature_matcher_wrapper(db_gt, query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, verbose = True, random_limit = random_no)
        # all of them as in first publiation (should be around 800 for each image)
        vanillia_matches = feature_matcher_wrapper(db_gt, query_images_names, train_descriptors_live, points3D_xyz_live, ratio_test_val, verbose = True )

        # get the benchmark data here for random (top50) features and the 800 from previous publication - will return the average values for each image
        benchmarks_iters = 15
        inlers_no, outliers, iterations, time, trans_errors_overall, rot_errors_overall = benchmark(benchmarks_iters, ransac, vanillia_matches, query_images_names, K, query_images_ground_truth_poses, scale, verbose=True)
        import pdb
        pdb.set_trace()

        # get the baseline matches and random matches here and pick a variable number

        # keys = list(logs.keys())
        # print("Starting training; got log keys: {}".format(keys))

    # def on_train_end(self, logs=None):
    #     keys = list(logs.keys())
    #     print("Stop training; got log keys: {}".format(keys))
    #
    # def on_epoch_begin(self, epoch, logs=None):
    #     keys = list(logs.keys())
    #     print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):

        import pdb
        pdb.set_trace()

        # 1 - get the gt poses
        # 2 - predict matchable features using current model (use those only for matching)
        # 3 - use all features for matching
        # 4 - use random features for matching
        # 5 - get inliers, ouliers, iterations, time (including prediction time), t_error, r_error mean for all epoch_images (or test - not actually test though)
        # 6 - compare the errors from 2,3 and 4 against the gt poses from COLMAP
        # 7 - pass the numbers to tensorboard

        # import pdb
        # pdb.set_trace()
        # writer = tf.summary.create_file_writer(self.tb_callback.log_dir) #self.tb_callback.writer
        # writer = tf.summary.create_file_writer(self.tb_callback.log_dir)
        #
        # writer = tf.summary.create_file_writer("/tmp/mylogs/eager")

        with self.writer.as_default():
            for step in range(100):
                # other model code would go here
                tf.summary.scalar("my_metric", 0.5, step=step)

        self.writer.flush()

        # import pdb
        # pdb.set_trace()
        #
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