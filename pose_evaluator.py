# This file will calculate the pose comaprison between the ones
# I calculated and the ones from COLMAP
# NOTE: run after ransac_comparison.py
import numpy as np
from query_image import read_images_binary, get_query_image_global_pose_new_model

def pose_evaluate(features_no):

    # images to the complete model containing all the query images (localised_images) + base images (ones used for SFM)
    complete_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/images.bin"
    complete_model_all_images = read_images_binary(complete_model_images_path)

    # load localised images names - This are from COLMAP
    localised_images = []
    path_to_query_images_file = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/"+features_no+"/images_localised_no_base.txt"
    with open(path_to_query_images_file) as f:
        localised_images = f.readlines()
    localised_images = [x.strip() for x in localised_images]

    exp_decay_rates_index = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for exp_decay_rate_index in exp_decay_rates_index:

        # TODO: why not using matches_base ?!?!! and comparing to that ? - need to look in ransac_comparison
        matches_all = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/"+features_no+"/matches_all_exp_dec_val_"+exp_decay_rate_index+".npy")

        print("Running for exponential decay value: 0." + exp_decay_rate_index)

        # these will hold the errors
        vanilla_ransac_results_t = []
        vanilla_ransac_results_a = []
        modified_ransac_results_t = []
        modified_ransac_results_a = []

        #  my poses calculated with my DM function
        vanilla_ransac_images_pose = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/2k/vanilla_ransac_images_pose_"+exp_decay_rate_index+".npy")
        modified_ransac_images_pose = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/2k/modified_ransac_images_pose_"+exp_decay_rate_index+".npy")

        for image in localised_images:
            if(matches_all.item()[image].shape[0] >= 4):
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

                # NOTE: arccos returns radians
                a_v = np.arccos((np.trace(np.dot(np.linalg.inv(pose_gt_R), v_r_pose_R)) - 1) / 2)
                vanilla_ransac_results_a.append(a_v)
                a_m = np.arccos((np.trace(np.dot(np.linalg.inv(pose_gt_R), m_r_pose_R)) - 1) / 2)
                modified_ransac_results_a.append(a_m)
            else:
                print(image + " has less than 4 matches..")

        print("For exponential decay value: 0."+ exp_decay_rate_index + " results are:")

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