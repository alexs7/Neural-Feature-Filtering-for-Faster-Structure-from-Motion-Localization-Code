# Run this to get the avg of the 3D desc of a point same order as in points3D
# The same file for base and live model can be find in ExMaps codebase (similar code)
# This is only for gt models and not the MnM paper data.
# For MnM refer to get_points_3D_mean_desc_ml_mnm.py

import os
import sys
import numpy as np
from tqdm import tqdm
from database import COLMAPDatabase
from parameters import Parameters
from point3D_loader import read_points3d_default, index_dict

SIZE = 129
def get_desc_avg(points3D, db):
    points_mean_descs_ids = np.empty([len(points3D.keys()), SIZE])

    point3D_vm_col_idx = 0
    for k,v in tqdm(points3D.items()):
        point_id = v.id
        points_image_ids = points3D[point_id].image_ids #COLMAP adds the image twice some times.
        points3D_descs = np.empty([len(points_image_ids), 128])
        # Loop through the points' image ids and check if it is seen by any image_ids
        # If it is seen then get the desc for each id.
        for k in range(len(points_image_ids)):
            id = points_image_ids[k]
            data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(id) + "'")
            data = COLMAPDatabase.blob_to_array(data.fetchone()[0], np.uint8)
            descs_rows = int(np.shape(data)[0] / 128)
            descs = data.reshape([descs_rows, 128]) #descs for the whole image
            keypoint_index = points3D[point_id].point2D_idxs[k]
            desc = descs[keypoint_index] #keypoints and descs are ordered the same (so I use the point2D_idxs to index descs )
            desc = desc.reshape(1, 128) #this is the desc of keypoint with index, keypoint_index, from image with id, id.
            points3D_descs[k] = desc

        # adding and calculating the mean here!
        mean = points3D_descs.mean(axis=0)
        points_mean_descs_ids[point3D_vm_col_idx] = np.append(mean, point_id).reshape(1,SIZE)
        point3D_vm_col_idx += 1
    return points_mean_descs_ids

root_path = "/media/iNicosiaData/engd_data/"
dataset = sys.argv[1] #HGE, CAB, LIN (or Other for CMU, retail shop)
print("Getting gt avg descs (not the MnM ones)")

if(dataset == "HGE" or dataset == "CAB" or dataset == "LIN"):
    path = os.path.join(root_path, f"lamar/{dataset}_colmap_model/")
    parameters = Parameters(path)

    gt_db = COLMAPDatabase.connect(parameters.gt_db_path)
    gt_model_points3D = read_points3d_default(parameters.gt_model_points3D_path)

    avgs_gt = get_desc_avg(gt_model_points3D, gt_db)
    np.save(parameters.avg_descs_gt_path, avgs_gt)

if(dataset == "CMU"):
    if(len(sys.argv) > 2):
        slices_names = [sys.argv[2]]
    else: # do all slices
        slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                        "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    for slice_name in slices_names:
        # overwrite paths
        path = os.path.join(root_path, f"cmu/{slice_name}/exmaps_data/")
        parameters = Parameters(path)

        gt_db = COLMAPDatabase.connect(parameters.gt_db_path)
        gt_model_points3D = read_points3d_default(parameters.gt_model_points3D_path)

        avgs_gt = get_desc_avg(gt_model_points3D, gt_db)
        np.save(parameters.avg_descs_gt_path, avgs_gt)

if(dataset == "RetailShop"):
    path = os.path.join(root_path, f"retail_shop/slice1/")
    parameters = Parameters(path)

    gt_db = COLMAPDatabase.connect(parameters.gt_db_path)
    gt_model_points3D = read_points3d_default(parameters.gt_model_points3D_path)

    avgs_gt = get_desc_avg(gt_model_points3D, gt_db)
    np.save(parameters.avg_descs_gt_path, avgs_gt)
