# run this to get the avg of the 3D desc of a point same order as in points3D
# be careful that you can get the base model's avg descs or the complete's model descs

# the idea here is that a point is seen by the base model images and complete model images
# obviously the complete model images number > base model images number for a point

import sqlite3
import numpy as np
import sys

from database import COLMAPDatabase, blob_to_array
from point3D_loader import read_points3d_default
from query_image import read_images_binary, image_localised

def get_desc_avg(features_no, out_path_base, out_path_all):

    print("-- Averaging features_no " + features_no + " --")

    db = COLMAPDatabase.connect("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/database.db")

    # by "complete model" I mean all the frames from future sessions localised in the base model (28/03) and the base model..
    # NOTE: I don't think it matters here to use features_no since the points3D and images are the same no matter what features_no was used in COLMAP. The database.db changes though..
    complete_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/images.bin"
    complete_model_all_images = read_images_binary(complete_model_images_path)
    complete_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/points3D.bin"
    points3D = read_points3d_default(complete_model_points3D_path) # base model's 3D points (same length as complete as we do not add points when localising new points, but different image_ds for each point)

    # create points id and index relationship
    point3D_index = 0
    points3D_indexing = {}
    for key, value in points3D.items():
        points3D_indexing[point3D_index] = value.id
        point3D_index = point3D_index + 1

    # get base images
    base_images_names = []
    with open("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/base_images.txt") as f:
        base_images_names = f.readlines()
    base_images_names = [x.strip() for x in base_images_names]

    base_images_ids = []
    for name in base_images_names:
        id = image_localised(name, complete_model_all_images)
        base_images_ids.append(id)

    print("base_images_ids size " + str(len(base_images_ids)))

    # get base + query images (query images are the future session images = all)
    query_images_names = []
    with open("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_name.txt") as f:
        query_images_names = f.readlines()
    query_images_names = [x.strip() for x in query_images_names]

    query_images_ids = []
    for name in query_images_names:
        id = image_localised(name, complete_model_all_images)
        query_images_ids.append(id) #NOTE: This will contain None values too!

    print("Localised and NonLocalised query_images_ids size: " + str(len(query_images_ids)))

    # This is used to check if a point3D was seen by an image in the base model - if so add that desc to the , points_mean_descs_base
    # it is also used to check if a point3D was seen by an image in the base model OR future sessions (i.e query image) - if so add that desc to points_mean_descs_all
    all_images_ids = base_images_ids + query_images_ids
    print("All_images_ids size: " + str(len(all_images_ids)))

    print("Getting the avg descs for the base and all (base + query) points' images")
    points_mean_descs_all = np.empty([0, 128])
    points_mean_descs_base = np.empty([0, 128])
    for i in range(0,len(points3D)):
        print("Doing point " + str(i+1) + "/" + str(len(points3D)), end="\r")
        point_id = points3D_indexing[i]
        points3D_descs_all = np.empty([0, 128])
        points3D_descs_base = np.empty([0, 128])
        # Loop through the points' image ids and check if it is seen by any base_images and all_images
        # If it is seen then get the descs for each id. len(points3D_descs_all) should be larger than len(points3D_descs_base) - always
        for k in range(len(points3D[point_id].image_ids)): #unique here doesn't really matter
            id = points3D[point_id].image_ids[k]
            # check if the point is viewed by the current base image
            if(id in base_images_ids):
                data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(id) + "'")
                data = blob_to_array(data.fetchone()[0], np.uint8)
                descs_rows = int(np.shape(data)[0] / 128)
                descs = data.reshape([descs_rows, 128])
                desc = descs[points3D[point_id].point2D_idxs[k]]
                desc = desc.reshape(1, 128)
                points3D_descs_base = np.r_[points3D_descs_base, desc]
            if (id in all_images_ids):
                data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(id) + "'")
                data = blob_to_array(data.fetchone()[0], np.uint8)
                descs_rows = int(np.shape(data)[0] / 128)
                descs = data.reshape([descs_rows, 128])
                desc = descs[points3D[point_id].point2D_idxs[k]]
                desc = desc.reshape(1, 128)
                points3D_descs_all = np.r_[points3D_descs_all, desc]
        if(len(points3D_descs_base) > len(points3D_descs_all)):
            raise Exception("points3D_descs_base size is larger than points3D_descs_all !?")
        # adding and calulating the mean here!
        points_mean_descs_base = np.r_[points_mean_descs_base, points3D_descs_base.mean(axis=0).reshape(1,128)]
        points_mean_descs_all = np.r_[points_mean_descs_all, points3D_descs_all.mean(axis=0).reshape(1,128)]

    print("\n Saving data...")
    # folder are created manually..
    np.save(out_path_base, points_mean_descs_base)
    np.save(out_path_all, points_mean_descs_all)

def get_desc_avg_with_session_weights(features_no, exponential_decay_value, out_path):

    print("-- Averaging features_no " + features_no + " and exponential_decay_rate "+ str(exponential_decay_value) +" --")

    # This contains weights for all base and query images
    session_weight_per_image = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/" + features_no + "/session_weight_per_image_" + str(exponential_decay_value) + ".npy")

    db = COLMAPDatabase.connect("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/database.db")

    # by "complete model" I mean all the frames from future sessions localised in the base model (28/03) and the base model..
    # NOTE: I don't think is matters here (for image.bin) to use features_no as the points3D and images are the same no matter what features_no was used in COLMAP. The database.db changes though..
    complete_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/images.bin"
    complete_model_all_images = read_images_binary(complete_model_images_path)
    complete_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/points3D.bin"
    points3D = read_points3d_default(complete_model_points3D_path) # base model's 3D points (same length as complete as we do not add points when localising new points, but different image_ds for each point)

    # create points id and index relationship
    point3D_index = 0
    points3D_indexing = {}
    for key, value in points3D.items():
        points3D_indexing[point3D_index] = value.id
        point3D_index = point3D_index + 1

    # get base images
    base_images_names = []
    with open("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/base_images.txt") as f:
        base_images_names = f.readlines()
    base_images_names = [x.strip() for x in base_images_names]

    base_images_ids = []
    for name in base_images_names:
        id = image_localised(name, complete_model_all_images)
        base_images_ids.append(id)

    print("base_images_ids size " + str(len(base_images_ids)))

    # get base + query images (query images are the future session images = all)
    query_images_names = []
    with open("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_name.txt") as f:
        query_images_names = f.readlines()
    query_images_names = [x.strip() for x in query_images_names]

    query_images_ids = []
    for name in query_images_names:
        id = image_localised(name, complete_model_all_images)
        query_images_ids.append(id) #NOTE: This will contain None values too!

    print("Localised and NonLocalised query_images_ids size: " + str(len(query_images_ids)))

    all_images_ids = base_images_ids + query_images_ids
    print("All_images_ids size: " + str(len(all_images_ids)))

    print("Getting the weighted descs for all (base + query) points' images")
    points_weighted_descs = np.empty([0, 128])
    for i in range(0,len(points3D)):
        print("Doing point " + str(i+1) + "/" + str(len(points3D)), end="\r")
        point_id = points3D_indexing[i]
        points3D_descs = np.empty([0,128])
        all_points_images_weights = []
        # Loop through the points' image ids and check if it is seen by all_images
        # If it is seen then get the descs for each id
        for k in range(len(points3D[point_id].image_ids)): #unique here doesn't really matter
            id = points3D[point_id].image_ids[k]
            if (id in all_images_ids):
                data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(id) + "'")
                data = blob_to_array(data.fetchone()[0], np.uint8)
                descs_rows = int(np.shape(data)[0] / 128)
                descs = data.reshape([descs_rows, 128])
                desc = descs[points3D[point_id].point2D_idxs[k]]
                desc = desc.reshape(1, 128)
                # collect descs for current point
                points3D_descs = np.r_[points3D_descs, desc]
                # collect weights
                image_name = db.execute("SELECT name FROM images WHERE image_id = " + "'" + str(id) + "'").fetchone()[0]
                weight = session_weight_per_image.item()[image_name]
                all_points_images_weights.append(weight)

        all_points_images_weights = all_points_images_weights / np.sum(all_points_images_weights)
        # points3D_descs and all_points_images_weights are in the same order
        points3D_descs = np.multiply(points3D_descs, all_points_images_weights[:, np.newaxis])
        # adding and calulating the mean here!
        points_weighted_descs = np.r_[points_weighted_descs, points3D_descs.sum(axis=0).reshape(1,128)]

    print("\n Saving data...")
    # folder are created manually..
    np.save(out_path, points_weighted_descs)

# colmap_features_no can be "2k", "1k", "0.5k", "0.25k"
# exponential_decay can be any of 0.1 to 0.9
# exponential_decay can be any of 0.1 to 0.9
features_no = "1k"
exponential_decay_value = 0.5

print("Running vanilla get 3D descs avg...")
out_path_base = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/"+features_no+"/points_mean_descs_base.npy"
out_path_all = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/"+features_no+"/points_mean_descs_all.npy"
get_desc_avg("1k", out_path_base, out_path_all)

out_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/"+features_no+"/points_weighted_descs_all_with_extra_data_"+str(exponential_decay_value)+".npy"
print("Running get 3D descs avg with session weights...")
get_desc_avg_with_session_weights("1k", 0.5, out_path)