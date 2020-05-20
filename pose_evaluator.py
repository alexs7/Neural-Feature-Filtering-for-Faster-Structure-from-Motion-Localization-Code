# This file will calculate the pose comaprison between the ones
# I calculated and the ones from COLMAP
import numpy as np
from query_image import read_images_binary, get_query_image_global_pose_new_model

#  my poses calculated with my DM function
vanilla_ransac_images_pose = np.load('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/vanilla_ransac_images_pose.npy')
modified_ransac_images_pose = np.load('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/modified_ransac_images_pose.npy')

# images to the complete model containing all the query images (localised_images) + base images (ones used for SFM)
complete_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/images.bin"
complete_model_all_images = read_images_binary(complete_model_images_path)

# load localised images names
localised_images = []
path_to_query_images_file = '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/images_localised.txt'
with open(path_to_query_images_file) as f:
    localised_images = f.readlines()
localised_images = [x.strip() for x in localised_images]

vanilla_ransac_results_t = []
vanilla_ransac_results_a = []
modified_ransac_results_t = []
modified_ransac_results_a = []
for image in localised_images:
    v_r_pose = vanilla_ransac_images_pose.item()[image]
    m_r_pose = modified_ransac_images_pose.item()[image]
    pose_gt = get_query_image_global_pose_new_model(image, complete_model_all_images)

    # translations errors
    v_r_pose_t = v_r_pose['Rt'][0:3,3]
    m_r_pose_t = m_r_pose['Rt'][0:3,3]
    pose_gt_t = pose_gt[0:3, 3]

    dist1 = np.linalg.norm(v_r_pose_t - pose_gt_t)
    vanilla_ransac_results_t.append(dist1)
    dist2 = np.linalg.norm(m_r_pose_t - pose_gt_t)
    modified_ransac_results_t.append(dist2)

    # rotations errors
    # from paper: Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions
    v_r_pose_R = v_r_pose['Rt'][0:3, 0:3]
    m_r_pose_R = m_r_pose['Rt'][0:3, 0:3]
    pose_gt_R = pose_gt[0:3, 0:3]

    # arccos returns radians
    a_v = np.arccos((np.trace(np.dot(np.linalg.inv(pose_gt_R), v_r_pose_R)) - 1) / 2)
    vanilla_ransac_results_a.append(a_v)
    a_m = np.arccos((np.trace(np.dot(np.linalg.inv(pose_gt_R), m_r_pose_R)) - 1) / 2)
    modified_ransac_results_a.append(a_m)

print("Averaged Errors Translations")
print("     Vanilla RANSAC: " + str(np.mean(vanilla_ransac_results_t)))
print("     Modified RANSAC: " + str(np.mean(modified_ransac_results_t)))

print("Averaged Errors Rotations")
print("     Vanilla RANSAC: " + str(np.mean(vanilla_ransac_results_a)))
print("     Modified RANSAC: " + str(np.mean(modified_ransac_results_a)))

print("Saving Data..")
np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/vanilla_ransac_results_t.npy", vanilla_ransac_results_t)
np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/modified_ransac_results_t.npy", modified_ransac_results_t)
np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/vanilla_ransac_results_a.npy", vanilla_ransac_results_a)
np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/modified_ransac_results_a.npy", modified_ransac_results_a)