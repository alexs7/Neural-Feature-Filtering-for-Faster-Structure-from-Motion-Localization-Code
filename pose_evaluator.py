# This file will calculate the pose comaprison between the ones
# I calculated and the ones from COLMAP
# NOTE: run after ransac_comparison.py

import numpy as np
from query_image import read_images_binary, get_query_image_global_pose_new_model, load_images_from_text_file

def pose_evaluate(features_no, exponential_decay_value,
                  matches_path, vanillia_data_path, modified_data_path,
                  save_path_vanillia_trans, save_path_vanillia_angles,
                  save_path_modified_trans, save_path_modified_angles):

    print("-- Doing features_no " + features_no + " --")

    # images to the complete model containing all the query images (localised_images) + base images (ones used for SFM)
    complete_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/images.bin"
    complete_model_all_images = read_images_binary(complete_model_images_path)

    # load localised images names - This are from COLMAP
    localised_images = load_images_from_text_file("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/images_localised_and_not_localised/" + features_no +"/images_localised.txt")
    # of course base images will be localised..
    base_images = load_images_from_text_file("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/base_images.txt")
    # Now, get localised images from the query images only. Not the base images.
    localised_query_images_only = []
    for image in localised_images:
        if (image not in base_images):
            localised_query_images_only.append(image)

    # TODO: why not using matches_base ?!?!! and comparing to that ? - look in ransac_comparison for explanation
    matches_all = np.load(matches_path)
    vanilla_images_pose = np.load(vanillia_data_path)
    modified_images_pose = np.load(modified_data_path)

    print("Running for exponential decay value: " + str(exponential_decay_value))

    # these will hold the errors
    vanilla_ransac_results_t = []
    vanilla_ransac_results_a = []
    modified_ransac_results_t = []
    modified_ransac_results_a = []

    for image in localised_query_images_only:

        if(matches_all.item()[image].shape[0] >= 4):

            v_r_pose = vanilla_images_pose.item()[image]
            m_r_pose = modified_images_pose.item()[image]
            pose_gt = get_query_image_global_pose_new_model(image, complete_model_all_images)

            # camera center errors
            v_r_pose_cam_c = v_r_pose['Rt'][0:3,0:3].transpose().dot(v_r_pose['Rt'][0:3,3])
            m_r_pose_cam_c = m_r_pose['Rt'][0:3,0:3].transpose().dot(m_r_pose['Rt'][0:3,3])
            pose_gt_cam_c = pose_gt[0:3,0:3].transpose().dot(pose_gt[0:3,3])

            dist1 = np.linalg.norm(v_r_pose_cam_c - pose_gt_cam_c)
            vanilla_ransac_results_t.append(dist1)

            dist2 = np.linalg.norm(m_r_pose_cam_c - pose_gt_cam_c)
            modified_ransac_results_t.append(dist2)

            # rotations errors
            # from paper: Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions
            v_r_pose_R = v_r_pose['Rt'][0:3, 0:3]
            m_r_pose_R = m_r_pose['Rt'][0:3, 0:3]
            pose_gt_R = pose_gt[0:3, 0:3]

            # NOTE: arccos returns radians - but I convert it to angles
            a_v = np.arccos((np.trace(np.dot(np.linalg.inv(pose_gt_R), v_r_pose_R)) - 1) / 2)
            vanilla_ransac_results_a.append(np.degrees(a_v))

            a_m = np.arccos((np.trace(np.dot(np.linalg.inv(pose_gt_R), m_r_pose_R)) - 1) / 2)
            modified_ransac_results_a.append(np.degrees(a_m))
        else:
            print(image + " has less than 4 matches..")

    print("Averaged Errors Translations")
    print("     Vanilla: " + str(np.mean(vanilla_ransac_results_t)))
    print("     Modified: " + str(np.mean(modified_ransac_results_t)))

    print("Averaged Errors Rotations (Degrees)")
    print("     Vanilla: " + str(np.mean(vanilla_ransac_results_a)))
    print("     Modified: " + str(np.mean(modified_ransac_results_a)))

    print("Saving Data..")
    np.save(save_path_vanillia_trans, vanilla_ransac_results_t)
    np.save(save_path_vanillia_angles, vanilla_ransac_results_a)

    np.save(save_path_modified_trans, modified_ransac_results_t)
    np.save(save_path_modified_angles, modified_ransac_results_a)

# colmap_features_no can be "2k", "1k", "0.5k", "0.25k"
# exponential_decay can be any of 0.1 to 0.9
features_no = "1k"
exponential_decay_value = 0.5

matches_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/"+features_no+"/matches_all.npy"
vanillia_data_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/vanilla_images_pose_"+str(exponential_decay_value)+".npy"
modified_data_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/modified_images_pose_"+str(exponential_decay_value)+".npy"

save_path_vanillia_trans = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/vanillia_results_t_"+features_no+"_"+str(exponential_decay_value)+".npy"
save_path_vanillia_angles = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/vanillia_results_a_"+features_no+"_"+str(exponential_decay_value)+".npy"
save_path_modified_trans = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/modified_results_t_"+features_no+"_"+str(exponential_decay_value)+".npy"
save_path_modified_angles = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/modified_results_a_"+features_no+"_"+str(exponential_decay_value)+".npy"

print("Evaluating Poses")
pose_evaluate("1k", 0.5, matches_path, vanillia_data_path, modified_data_path,
                  save_path_vanillia_trans, save_path_vanillia_angles,
                  save_path_modified_trans, save_path_modified_angles)

