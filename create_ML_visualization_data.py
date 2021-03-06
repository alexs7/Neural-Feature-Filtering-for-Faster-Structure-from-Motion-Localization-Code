import os
import numpy as np
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
from tensorflow import keras
import colmap
from database import COLMAPDatabase
from parameters import Parameters
from query_image import get_image_id, get_keypoints_xy, get_queryDescriptors
from point3D_loader import read_points3d_default, index_dict_reverse

# UPDATE: as of 04/06/2021 This is not used any more , but kept around if needed to quickly visualise things in ThreeJS

# This file will create the visualization data to view in threejs or any other viewer you create.
# For images and pointclouds
# Even though it is called visual.db, it is not used for testing ML models as unseen data.

# Command:
# python3 create_ML_visualization_data.py
# /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/visual.db
# /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/visual_images/2020-06-22/
# /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/images_list.txt
# /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/model/
# /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/visual_data/images/
# /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/visual_data/points/points_predictions.db
# /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/visual_data/points/
# /home/alex/fullpipeline/colmap_data/Coop_data/slice1/
# oneliner: python3 create_ML_visualization_data.py /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/visual.db /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/visual_images/2020-06-22/ /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/images_list.txt /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/model/ /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/visual_data/images/ /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/visual_data/points/points_predictions.db /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/visual_data/points/ /home/alex/fullpipeline/colmap_data/Coop_data/slice1/

# test_db.db will be used to add data, so delete it before running this script
test_db_path = sys.argv[1]
images_dir = sys.argv[2]
image_list_file = sys.argv[3]
model_path = sys.argv[4]
save_path_images = sys.argv[5]
db_path_points = sys.argv[6]
save_path_points = sys.argv[7]
base_path = sys.argv[8] # example: "/home/alex/fullpipeline/colmap_data/CMU_data/slice1/" #trailing "/"

# make sure the templates_ini/feature_extractions file are the same between Mobile-Pose.. and fullpipeline
colmap.feature_extractor(test_db_path, images_dir, image_list_file, query=True)
db = COLMAPDatabase.connect(test_db_path)
db_points_preds = COLMAPDatabase.create_db_for_visual_data(db_path_points)

with open(image_list_file) as f:
    query_images = f.readlines()
query_images = [x.strip() for x in query_images]

model = keras.models.load_model(model_path)

# Images
for i in range(len(query_images)):
    print("Image no: " + str(i) + "/" + str(len(query_images)), end="\r")
    q_img = query_images[i]
    image_id = get_image_id(db, q_img)
    # keypoints data
    keypoints_xy = get_keypoints_xy(db, image_id)
    queryDescriptors = get_queryDescriptors(db, image_id)

    predictions = model.predict(queryDescriptors)

    data = np.concatenate((keypoints_xy, predictions), axis=1)
    data = data[data[:, 2].argsort()[::-1]]

    np.savetxt(save_path_images + q_img.split(".")[0]+".txt", data)

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
no = -1
for k,v in points3D.items():
    no += 1
    print("Point no: " + str(no) + "/" + str(len(points3D)), end="\r")
    index = points3D_indexing[v.id]
    score = points3D_per_image_decay_scores[index]
    avg_sift_vector = points3D_avg_sift_desc[index]
    pred_score = model.predict(avg_sift_vector.reshape(1, 128))
    pred_score_float64 = pred_score.astype(np.float64)[0][0] #for db reasons
    xyz = v.xyz
    db_points_preds.execute("INSERT INTO data VALUES (?, ?, ?, ?)",
                            (COLMAPDatabase.array_to_blob(avg_sift_vector),) +
                            (pred_score_float64,) +
                            (score,) +
                            (COLMAPDatabase.array_to_blob(xyz),))
    # row = np.array([v.xyz[0], v.xyz[1], v.xyz[2], pred_score, score]).reshape([1,5])
    # row = np.c_[row, avg_sift_vector.reshape([1, 128])]
    # points3D_xyz_score_sift = np.r_[points3D_xyz_score_sift, row]

db_points_preds.commit()

points_preds_and_gt = db_points_preds.execute("SELECT pred_score, score, xyz FROM data").fetchall()
xyzs = (COLMAPDatabase.blob_to_array(row[2] , np.float64) for row in points_preds_and_gt)
xyzs = np.array(list(xyzs))
pred_scores = (row[0] for row in points_preds_and_gt)
pred_scores = np.array(list(pred_scores))
scores = (row[1] for row in points_preds_and_gt)
scores = np.array(list(scores))
visual_points_data = np.c_[xyzs, pred_scores, scores]

# sort by score
points3D_sorted_by_pred_score = visual_points_data[visual_points_data[:,3].argsort()[::-1]]
points3D_sorted_by_score = visual_points_data[visual_points_data[:,4].argsort()[::-1]]

np.savetxt(save_path_points + "points3D_sorted_by_pred_score.txt", points3D_sorted_by_pred_score)
np.savetxt(save_path_points + "points3D_sorted_by_score.txt", points3D_sorted_by_score)

print("Done!")