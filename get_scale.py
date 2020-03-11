import random
import numpy as np
from evaluator import get_sequence_all
from evaluator import get_ARCore_displayOrientedPose
from query_image import get_query_image_global_pose
from scipy.spatial import procrustes

query_dir = '/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/images/data_ar'
sequence = get_sequence_all(query_dir)
iterations = 100

def calc_scale(use_procrustes = False):
    if(use_procrustes == False):
        print("Calculating scale..")
        scales = []
        for i in range(iterations):
            #random poses
            start = random.choice(sequence)
            end = random.choice(sequence)

            #get 2 poses in COLMAP
            colmap_pose_start = get_query_image_global_pose("frame_"+start+".jpg")
            colmap_pose_end = get_query_image_global_pose("frame_"+end+".jpg")

            colmap_pose_start_pos = colmap_pose_start[0:3, 3]
            colmap_pose_end_pos = colmap_pose_end[0:3, 3]

            dist_colmap = np.linalg.norm(colmap_pose_start_pos - colmap_pose_end_pos)

            #get 2 poses in ARCore
            arCore_start = np.linalg.inv(get_ARCore_displayOrientedPose(query_dir, start))
            arCore_end = np.linalg.inv(get_ARCore_displayOrientedPose(query_dir, end))

            arCore_start_pos = arCore_start[0:3, 3]
            arCore_end_pos = arCore_end[0:3, 3]

            dist_arCore = np.linalg.norm(arCore_start_pos - arCore_end_pos)

            scale = dist_arCore / dist_colmap
            scales.append(scale)
            scales = np.array(scales)
            return np.mean(scales)
    else:
        print("Generating Procrustes data for Matlab..")
        colmap_locations = []
        arCore_locations = []
        for i in range(len(sequence)):
            no = sequence[i]
            colmap_pose = get_query_image_global_pose("frame_" + no + ".jpg")
            arCore_pose = np.linalg.inv(get_ARCore_displayOrientedPose(query_dir, no))

            colmap_trans = colmap_pose[0:3, 3]
            arCore_trans = arCore_pose[0:3, 3]

            colmap_locations.append(colmap_trans)
            arCore_locations.append(arCore_trans)

        np.savetxt('/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/colmap_locations_procrustes.txt',colmap_locations)
        np.savetxt('/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/arcore_locations_procrustes.txt',arCore_locations)



