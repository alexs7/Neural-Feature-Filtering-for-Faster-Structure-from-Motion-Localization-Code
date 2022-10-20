# The difference form "get_points_3D_mean_desc_single_model.py" or "get_points_3D_mean_descs.py" as it returns the SIFT descs and coordinate x,y,z points.
# And also I do not normalise the descs here.

# one liner (for weatherwax server, ogg etc, or the docker in them)
# using relative paths here (CMU or Coop)
#  arguments
# 1 - live db
# 2 - live images
# 3 - live points3D
# 4 - output filename
# python3 get_points_3D_mean_desc_single_model_ml.py colmap_data/CMU_data/slice3/live/database.db colmap_data/CMU_data/slice3/live/model/images.bin colmap_data/CMU_data/slice3/live/model/points3D.bin colmap_data/CMU_data/slice3/ML_data/avg_descs_xyz_ml.npy

# 21/10/2022: For OpenCV models (MnM paper) use "output_opencv_sift_model" in the command

import sys
import numpy as np
from database import COLMAPDatabase
from point3D_loader import read_points3d_default
from query_image import read_images_binary

def get_point_info(points3D, db):
    no = 0
    points_info = np.empty([0, 131]) #(SIFT + xyz)

    for k, v in points3D.items():
        no += 1
        point_id = v.id
        points3D_descs = np.empty([0, 128])
        points_image_ids = points3D[point_id].image_ids #COLMAP adds the image twice some times.
        # Loop through the points' image ids and check if it is seen by any image_ids
        # If it is seen then get the desc for each id.
        for k in range(len(points_image_ids)):
            print("Point: " + str(no) + ", Image: " + str(k), end="\r")
            id = points_image_ids[k]
            data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(id) + "'")
            data = COLMAPDatabase.blob_to_array(data.fetchone()[0], np.uint8)
            descs_rows = int(np.shape(data)[0] / 128)
            descs = data.reshape([descs_rows, 128]) #descs for the whole image
            keypoint_index = points3D[point_id].point2D_idxs[k]
            desc = descs[keypoint_index] #keypoints and descs are ordered the same (so I use the point2D_idxs to index descs )
            desc = desc.reshape(1, 128) #this is the desc of keypoint with index, keypoint_index, from image with id, id.
            points3D_descs = np.r_[points3D_descs, desc]

        # adding and calulating the mean here!
        mean = points3D_descs.mean(axis=0).reshape(1,128)
        row = np.append(mean, v.xyz)
        points_info = np.r_[points_info, row.reshape(1, 131)]
    return points_info

print()

db_path = sys.argv[1]
model_images_bin_path = sys.argv[2]
model_points3D_bin_path = sys.argv[3]
save_path = sys.argv[4]

db = COLMAPDatabase.connect(db_path)
images = read_images_binary(model_images_bin_path)
points3D = read_points3d_default(model_points3D_bin_path)

avgs = get_point_info(points3D, db)
np.save(save_path, avgs)


