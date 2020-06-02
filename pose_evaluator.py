# This file will calculate the pose comaprison between the ones
# I calculated and the ones from COLMAP
# NOTE: run after ransac_comparison.py

import numpy as np
from query_image import read_images_binary, get_query_image_global_pose_new_model, load_images_from_text_file

def pose_evaluate(features_no):

    print("-- Doing features_no " + features_no + " --")

    # images to the complete model containing all the query images (localised_images) + base images (ones used for SFM)
    complete_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/images.bin"
    complete_model_all_images = read_images_binary(complete_model_images_path)

    # load localised images names - This are from COLMAP
    localised_images = load_images_from_text_file("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/images_localised_and_not_localised/" + features_no +"/images_localised.txt")
    # of course base images will be localised..
    base_images = load_images_from_text_file("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/base_images.txt")

    exp_decay_rates_index = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for exp_decay_rate_index in exp_decay_rates_index:

        # TODO: why not using matches_base ?!?!! and comparing to that ? - look in ransac_comparison for explanation
        matches_all = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/"+features_no+"/matches_all.npy")

        print("Running for exponential decay value: 0." + exp_decay_rate_index)

        # these will hold the errors
        vanilla_ransac_results_t = []
        vanilla_ransac_results_a = []
        modified_ransac_results_t = []
        modified_ransac_results_a = []

        #  my poses calculated with my DM function and different RANSAC versions
        vanilla_ransac_images_pose = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/vanilla_ransac_images_pose_"+exp_decay_rate_index+".npy")
        modified_ransac_images_pose = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/modified_ransac_images_pose_"+exp_decay_rate_index+".npy")

        for image in localised_images:
            if(image not in base_images):
                if(matches_all.item()[image].shape[0] >= 4):
                    v_r_pose = vanilla_ransac_images_pose.item()[image]
                    m_r_pose = modified_ransac_images_pose.item()[image]
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
        print("     Vanilla RANSAC: " + str(np.mean(vanilla_ransac_results_t)))
        print("     Modified RANSAC: " + str(np.mean(modified_ransac_results_t)))

        print("Averaged Errors Rotations")
        print("     Vanilla RANSAC: " + str(np.mean(vanilla_ransac_results_a)))
        print("     Modified RANSAC: " + str(np.mean(modified_ransac_results_a)))

        print("Saving Data..")
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/vanilla_ransac_results_t_"+features_no+"_"+exp_decay_rate_index+".npy", vanilla_ransac_results_t)
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/modified_ransac_results_t_"+features_no+"_"+exp_decay_rate_index+".npy", modified_ransac_results_t)
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/vanilla_ransac_results_a_"+features_no+"_"+exp_decay_rate_index+".npy", vanilla_ransac_results_a)
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/modified_ransac_results_a_"+features_no+"_"+exp_decay_rate_index+".npy", modified_ransac_results_a)

    print("")

colmap_features_no = ["2k", "1k", "0.5k", "0.25k"]

# run for each no of features
for features_no in colmap_features_no:
    pose_evaluate(features_no)