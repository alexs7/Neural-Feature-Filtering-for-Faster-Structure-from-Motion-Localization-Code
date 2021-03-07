import os
import numpy as np
from tensorflow import keras
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import colmap
from database import COLMAPDatabase
from parameters import Parameters
from query_image import get_image_id, get_keypoints_xy, get_queryDescriptors
from point3D_loader import read_points3d_default, index_dict_reverse

# This file will create the visualization data to view in threejs or any other viewer you create.
# For images and pointclouds

# Command:
# python3 create_ML_visualization_data.py
# /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/test_db.db
# /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/test_images/2020-06-22/
# /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/images_list.txt
# /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/model/
# /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/visual_data/images/
# /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/visual_data/points/points3D_sorted_descending_heatmap_per_image.txt
# /home/alex/fullpipeline/colmap_data/Coop_data/slice1/
# oneliner: python3 create_ML_visualization_data.py /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/test_db.db /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/test_images/2020-06-22/ /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/images_list.txt /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/model/ /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/visual_data/images/ /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/visual_data/points/points3D_sorted_descending_heatmap_per_image.txt /home/alex/fullpipeline/colmap_data/Coop_data/slice1/

# test_db.db will be used to add data, so delete it before running this script
test_db_path = sys.argv[1]
images_dir = sys.argv[2]
image_list_file = sys.argv[3]
model_path = sys.argv[4]
save_path_images = sys.argv[5]
save_path_points = sys.argv[6]
base_path = sys.argv[7] # example: "/home/alex/fullpipeline/colmap_data/CMU_data/slice1/" #trailing "/"

# make sure the templates_ini/feature_extractions file are the same between Mobile-Pose.. and fullpipeline
colmap.feature_extractor(test_db_path, images_dir, image_list_file, query=True)
db = COLMAPDatabase.connect(test_db_path)

with open(image_list_file) as f:
    query_images = f.readlines()
query_images = [x.strip() for x in query_images]

model = keras.models.load_model(model_path)

# Images
# for i in range(len(query_images)):
#     q_img = query_images[i]
#     image_id = get_image_id(db, q_img)
#     # keypoints data
#     keypoints_xy = get_keypoints_xy(db, image_id)
#     queryDescriptors = get_queryDescriptors(db, image_id)
#
#     predictions = model.predict(queryDescriptors)
#
#     data = np.concatenate((keypoints_xy, predictions), axis=1)
#     data = data[data[:, 2].argsort()[::-1]]
#
#     np.savetxt(save_path_images + q_img.split(".")[0]+".txt", data)

# Points
parameters = Parameters(base_path)
# Load scores
points3D_per_image_decay_scores = np.load(parameters.per_image_decay_matrix_path)
points3D_per_image_decay_scores = points3D_per_image_decay_scores.sum(axis=0)
# min-max normalization (not dividing by sum()) - this is done because in training I do the same, so it is a way to compare the model's output with these scores
points3D_per_image_decay_scores = ( points3D_per_image_decay_scores - points3D_per_image_decay_scores.min() ) / ( points3D_per_image_decay_scores.max() - points3D_per_image_decay_scores.min() )
# read points
points3D = read_points3d_default(parameters.live_model_points3D_path) #Needs to be live model points, because of ids changing compared to base model ( because of colmap )
# load sift avgs for each point
points3D_avg_sift_desc = np.load(parameters.avg_descs_live_path)

total_dims = 133
points3D_xyz_score_sift = np.empty([0, total_dims])
points3D_indexing = index_dict_reverse(points3D)
for k,v in points3D.items():
    index = points3D_indexing[v.id]
    score = points3D_per_image_decay_scores[index]
    avg_sift_vector = points3D_avg_sift_desc[index]
    import pdb
    pdb.set_trace()
    row = np.array([v.xyz[0], v.xyz[1], v.xyz[2], score]).reshape([1,4])
    row = np.c_[row, avg_sift_vector.reshape([1, 128])]
    points3D_xyz_score_sift = np.r_[points3D_xyz_score_sift, row]

points3D_xyz_score_sift = points3D_xyz_score_sift[points3D_xyz_score_sift[:,3].argsort()[::-1]]
# sort points
np.savetxt(save_path_points, points3D_xyz_score_sift)