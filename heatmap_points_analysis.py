# this is to match sfm images descs to the
# base model and complete model as a benchmark
import sqlite3

import numpy as np
import sys
import cv2

from feature_matcher import feature_matcher_factory, FeatureMatcherTypes
from parameters import Parameters
from point3D_loader import read_points3d_default
from query_image import read_images_binary, image_localised

IS_PYTHON3 = sys.version_info[0] >= 3

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

db = COLMAPDatabase.connect("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/database.db")

# by "complete model" I mean all the frames from future sessions localised in the base model (28/03)
complete_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/images.bin"
complete_model_all_images = read_images_binary(complete_model_images_path)
complete_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/points3D.bin"
points3D = read_points3d_default(complete_model_points3D_path) # base model's 3D points

# create points id and index relationship
point3D_index = 0
points3D_indexing = {}
for key, value in points3D.items():
    points3D_indexing[point3D_index] = value.id
    point3D_index = point3D_index + 1

# images to use for testing
test_images = []
path_to_query_images_file = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_name.txt"
with open(path_to_query_images_file) as f:
    test_images = f.readlines()
test_images = [x.strip() for x in test_images]

print("Loading 3D Points mean descs")
# this creates the descs means for the base model and the complete model
trainDescriptors_all = np.loadtxt('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/points_mean_descs_all.txt')
trainDescriptors_base = np.loadtxt('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/points_mean_descs_base.txt')

results_base = []
results_all = []

matches_base = np.empty([0,5])
matches_all = np.empty([0,5])

images_not_localised = []
images_localised = []
#  go through all the test images and match their descs to the 3d points avg descs
for test_image in test_images:
    print("Doing image " + test_image)
    image_id = image_localised(test_image, complete_model_all_images)
    if(image_id != None):
        images_localised.append(test_image)
        image_id = str(image_id)
        # fetching the x,y,descs for that image
        query_image_keypoints_data = db.execute("SELECT data FROM keypoints WHERE image_id = " + "'" + image_id + "'")
        query_image_keypoints_data = query_image_keypoints_data.fetchone()[0]
        query_image_keypoints_data_cols = db.execute("SELECT cols FROM keypoints WHERE image_id = " + "'" + image_id + "'")
        query_image_keypoints_data_cols = int(query_image_keypoints_data_cols.fetchone()[0])
        query_image_keypoints_data = blob_to_array(query_image_keypoints_data, np.float32)
        query_image_keypoints_data_rows = int(np.shape(query_image_keypoints_data)[0] / query_image_keypoints_data_cols)
        query_image_keypoints_data = query_image_keypoints_data.reshape(query_image_keypoints_data_rows,query_image_keypoints_data_cols)
        query_image_keypoints_data_xy = query_image_keypoints_data[:, 0:2]

        query_image_descriptors_data = db.execute(
            "SELECT data FROM descriptors WHERE image_id = " + "'" + image_id + "'")
        query_image_descriptors_data = query_image_descriptors_data.fetchone()[0]
        query_image_descriptors_data = blob_to_array(query_image_descriptors_data, np.uint8)
        descs_rows = int(np.shape(query_image_descriptors_data)[0] / 128)
        query_image_descriptors_data = query_image_descriptors_data.reshape([descs_rows, 128])
        query_keypoints_xy_descriptors = np.concatenate((query_image_keypoints_data_xy, query_image_descriptors_data), axis=1)

        # once you have the test images descs now do feature matching here! - Matching on all and base descs means
        queryDescriptors = query_image_descriptors_data.astype(np.float32)
        trainDescriptors_all = trainDescriptors_all.astype(np.float32)
        trainDescriptors_base = trainDescriptors_base.astype(np.float32)

        matching_algo = FeatureMatcherTypes.FLANN  # or FeatureMatcherTypes.BF
        match_ratio_test = Parameters.kFeatureMatchRatioTest
        norm_type = cv2.NORM_L2
        cross_check = False
        matcher = feature_matcher_factory(norm_type, cross_check, match_ratio_test, matching_algo)
        good_matches_all = matcher.match(queryDescriptors, trainDescriptors_all)
        good_matches_base = matcher.match(queryDescriptors, trainDescriptors_base)

        results_all.append(len(good_matches_all[0]))
        results_base.append(len(good_matches_base[0]))

        # matches_all = ma
        # breakpoint()

        print("Found this many good matches (against complete model): " + str(len(good_matches_all[0])) + ", " + str(
            len(good_matches_all[1])))
        print("Found this many good matches (against base model): " + str(len(good_matches_base[0])) + ", " + str(
            len(good_matches_base[1])))
    else:
        print("Frame "+test_image+" not localised..")
        images_not_localised.append(test_image)


np.savetxt('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/results_all.txt', np.array([results_all]))
np.savetxt('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/results_base.txt', np.array([results_base]))

# also write the names for the graphs
with open('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/images_localised.txt', 'w') as f:
    for item in images_localised:
        f.write("%s\n" % item)

with open('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/images_not_localised.txt', 'w') as f:
    for item in images_not_localised:
        f.write("%s\n" % item)
