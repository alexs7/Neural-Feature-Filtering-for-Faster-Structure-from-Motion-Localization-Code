# run this to get the avg of the 3D desc of a point same order as in points3D
# be careful that you can get the base model's avg descs or the complete's model descs

# the idea here is that a point is seen by the base model images and complete model images
# obviously the complete model images number > base model images number for a point

import numpy as np
from database import COLMAPDatabase, blob_to_array
from point3D_loader import read_points3d_default, index_dict
from query_image import read_images_binary, get_images_names_bin, get_images_ids


def get_desc_avg(save_path, image_ids, points3D, db, exponential_decay_value=None, session_weight_per_image = None):

    do_weighted = exponential_decay_value == True and session_weight_per_image != None
    all_points_images_weights = [] # if(exponential_decay_value == True and session_weight_per_image != None)
    points_mean_descs = np.empty([0, 128])

    for k,v in points3D.items():
        point_id = v.id
        points3D_descs = np.empty([0, 128])
        points_image_ids = points3D[point_id].image_ids
        # Loop through the points' image ids and check if it is seen by any image_ids
        # If it is seen then get the desc for each id. len(points3D_descs_all) should be larger than len(points3D_descs_base) - always
        for k in range(len(points_image_ids)): #unique here doesn't really matter TODO: double  check that
            id = points3D[point_id].image_ids[k]
            # check if the point is viewed by the current base image
            if(id in image_ids):
                data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(id) + "'")
                data = blob_to_array(data.fetchone()[0], np.uint8)
                descs_rows = int(np.shape(data)[0] / 128)
                descs = data.reshape([descs_rows, 128])
                keypoint_index = points3D[point_id].point2D_idxs[k]
                desc = descs[keypoint_index] #keypoints and descs are ordred the same (so I use the point2D_idxs to index descs )
                desc = desc.reshape(1, 128)
                points3D_descs = np.r_[points3D_descs, desc]
                if(do_weighted):
                    # collect weights
                    image_name = db.execute("SELECT name FROM images WHERE image_id = " + "'" + str(id) + "'").fetchone()[0]
                    weight = session_weight_per_image.item()[image_name]
                    all_points_images_weights.append(weight)

        if(do_weighted):
            all_points_images_weights = all_points_images_weights / np.sum(all_points_images_weights)
            # points3D_descs and all_points_images_weights are in the same order
            points3D_descs = np.multiply(points3D_descs, all_points_images_weights[:, np.newaxis])

        # adding and calulating the mean here!
        points_mean_descs = np.r_[points_mean_descs, points3D_descs.mean(axis=0).reshape(1,128)]

    return points_mean_descs

# colmap_features_no can be "2k", "1k", "0.5k", "0.25k"
# exponential_decay can be any of 0.1 to 0.9
features_no = "1k"
exponential_decay_value = 0.5

# method get_desc_avg() will take as main arguments image names and a model that has base + query images localised (complete)
# for example if you pass base_model_images_names and the complete_model it will only average descs from base images.
# when you use the complete model you use the one that you localised all the queries against to, and complete_model_images_path also includes base images

db = COLMAPDatabase.connect("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/" + features_no + "/database.db")

base_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/coop_local/2020-03-28/coop_local/reconstruction/model/0/images.bin"
complete_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/" + features_no + "/images.bin"

complete_model_all_images = read_images_binary(complete_model_images_path)
complete_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/" + features_no + "/points3D.bin"
points3D = read_points3d_default(complete_model_points3D_path)  # base model's 3D points (same length as complete as we do not add points when localising new points, but different image_ids for each point)

print("-- Averaging features_no " + features_no + " --")

# 4 cases here:
# non-weighted, base | all
# weighted, base | all

base_images_names = get_images_names_bin(base_model_images_path)
all_images_names = get_images_names_bin(complete_model_images_path) #all = query + base images

base_images_ids = get_images_ids(base_images_names, complete_model_all_images) #there should not be any None' values here
all_images_ids = get_images_ids(all_images_names, complete_model_all_images) #there should not be any None' values here

print("Getting non-weighted descs")
save_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/"+features_no+"/avg_descs_base.npy"
avgs = get_desc_avg(save_path, base_images_ids, points3D,db)
np.save(save_path, avgs)

save_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/"+features_no+"/avg_descs_all.npy"
avgs = get_desc_avg(save_path, all_images_ids, points3D,db)
np.save(save_path, avgs)

print("Getting weighted descs")
session_weight_per_image = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/"+features_no+"/session_weight_per_image_" + str(exponential_decay_value) + ".npy")

save_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/"+features_no+"/avg_descs_base_weighted.npy"
avgs = get_desc_avg(save_path, base_images_ids, points3D,db)
np.save(save_path, avgs)

save_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/"+features_no+"/avg_descs_all_weighted.npy"
avgs = get_desc_avg(save_path, all_images_ids, points3D,db)
np.save(save_path, avgs)




