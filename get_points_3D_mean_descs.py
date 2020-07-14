# run this to get the avg of the 3D desc of a point same order as in points3D
# be careful that you can get the base model's avg descs or the live's model descs - depends on the points images ids

# the idea here is that a point is seen by the base model images and live model images
# obviously the live model images number > base model images number for a point

import numpy as np
from database import COLMAPDatabase
from parameters import Parameters
from point3D_loader import read_points3d_default, index_dict
from query_image import read_images_binary, get_images_ids, get_images_names_from_sessions_numbers

def get_desc_avg(image_ids, points3D, db):
    # Note: Look at this method this way: "I want to average descs of a 3D point that belong to certain images
    # (some from base or some from all, images) and not average all the 3D points descs."
    points_mean_descs = np.empty([0, 128])

    for k,v in points3D.items():
        point_id = v.id
        points3D_descs = np.empty([0, 128])
        points_image_ids = np.unique(points3D[point_id].image_ids) #This is because COLMAP adds the image twice some times.
        # Loop through the points' image ids and check if it is seen by any image_ids
        # If it is seen then get the desc for each id.
        for k in range(len(points_image_ids)):
            id = points3D[point_id].image_ids[k]
            # check if the point is viewed by the current image
            if(id in image_ids):
                data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(id) + "'")
                data = COLMAPDatabase.blob_to_array(data.fetchone()[0], np.uint8)
                descs_rows = int(np.shape(data)[0] / 128)
                descs = data.reshape([descs_rows, 128]) #descs for the whole image
                keypoint_index = points3D[point_id].point2D_idxs[k]
                desc = descs[keypoint_index] #keypoints and descs are ordered the same (so I use the point2D_idxs to index descs )
                desc = desc.reshape(1, 128) #this is the desc of keypoint with index, keypoint_index, from image with id, id.
                points3D_descs = np.r_[points3D_descs, desc]

        # adding and calulating the mean here!
        points_mean_descs = np.r_[points_mean_descs, points3D_descs.mean(axis=0).reshape(1,128)]
    return points_mean_descs

# colmap_features_no can be "2k", "1k", "0.5k", "0.25k"
# exponential_decay can be any of 0.1 to 0.9
features_no = "1k"
exponential_decay_value = 0.5

print("-- Averaging features_no " + features_no + " --")

# method get_desc_avg() will take as main arguments image names and a model that has base + session images localised (live)
# for example if you pass base_model_images_names and the live model it will only average descs from base images.

db_live_path = Parameters.live_db_path
db_live = COLMAPDatabase.connect(db_live_path)

db_base_path = Parameters.base_db_path
db_base = COLMAPDatabase.connect(db_base_path)

live_model_images_path = Parameters.live_model_images_path
live_model_images = read_images_binary(live_model_images_path)
live_model_points3D_path = Parameters.live_model_points3D_path
live_model_points3D = read_points3d_default(live_model_points3D_path)

#  no_images_per_session[0] is base images
no_images_per_session = Parameters.no_images_per_session

# 2 cases base and live images points3D descs
base_images_names = get_images_names_from_sessions_numbers([no_images_per_session[0]], db_base, live_model_images) # need to pass array here
all_images_names = get_images_names_from_sessions_numbers(no_images_per_session, db_live, live_model_images) #all = query + base images

base_images_ids = get_images_ids(base_images_names, live_model_images) #there should not be any None' values here
all_images_ids = get_images_ids(all_images_names, live_model_images) #there should not be any None' values here

# You will notice that I am using live_model_points3D in both cases, fetching avg features for the base images and the live images.
# This is because the live_model_points3D points' images_ids hold also ids of the live and base model images, since the live model is just the
# base model with extra images localised in it. You can use the base model for the base images but you need to make sure that the base model is exactly the
# same as the live model.

save_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/"+features_no+"/avg_descs_base.npy"
avgs = get_desc_avg(base_images_ids, live_model_points3D, db_base)
np.save(save_path, avgs)

save_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/"+features_no+"/avg_descs_live.npy"
avgs = get_desc_avg(all_images_ids, live_model_points3D, db_live)
np.save(save_path, avgs)



