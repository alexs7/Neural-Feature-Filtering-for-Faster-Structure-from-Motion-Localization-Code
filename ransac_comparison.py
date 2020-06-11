import numpy as np

from point3D_loader import read_points3d_default
from query_image import load_images_from_text_file, read_images_binary
from ransac import run_ransac, run_ransac_modified, run_prosac
import time

# This was used with the old modified RANSAC version
# get the "sub_distributions" for each matches set for each image - This will have to be relooked at!
# def get_sub_distribution(matches_for_image, distribution):
#     indices = matches_for_image[:, 5]
#     indices = indices.astype(int)
#     sub_distribution = distribution[0, indices]
#     sub_distribution = sub_distribution / np.sum(sub_distribution)
#     return sub_distribution

def run_comparison(features_no, exponential_decay_value, run_ransac, run_prosac, matches_path,
                   vanillia_data_path_save_poses, modified_data_path_save_poses,
                   vanillia_data_path_save_info, modified_data_path_save_info):

    print("-- Doing features_no " + features_no + " --")

    # load localised images names (including base ones)
    localised_images = load_images_from_text_file("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/images_localised_and_not_localised/" + features_no + "/images_localised.txt")
    # of course base images will be localised..
    base_images = load_images_from_text_file("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/base_images.txt")
    # Now, get localised images from the query images only. Not the base images.
    localised_query_images_only = []
    for image in localised_images:
        if(image not in base_images):
            localised_query_images_only.append(image)

    # Question: why not using matches_base ?!?!! and comparing to that ?
    # Answer: and compare what ? ransac will take less time with matches_base given the smaller number of all matches,
    # plus the get_sub_distribution might not make sense here as for base there are no future sessions yet.
    matches_all = np.load(matches_path)

    print("Running RANSAC/PROSAC.. for exponential decay of value: " + str(exponential_decay_value))

    #  this will hold inliers_no, ouliers_no, iterations, time for each image
    vanilla_data = np.empty([0, 4])
    vanilla_images_poses = {}

    modified_data = np.empty([0, 4])
    modified_images_poses = {}

    for i in range(len(localised_query_images_only)):
        image = localised_query_images_only[i]
        matches_for_image = matches_all.item()[image]
        print("Doing image " + str(i+1) + "/" + str(len(localised_query_images_only)) + ", " + image , end="\r")

        if(len(matches_for_image) >= 4):
            # vanilla
            start = time.time()
            inliers_no, ouliers_no, iterations, best_model = run_ransac(matches_for_image)
            end  = time.time()
            elapsed_time = end - start

            vanilla_images_poses[image] = best_model
            vanilla_data = np.r_[vanilla_data, np.array([inliers_no, ouliers_no, iterations, elapsed_time]).reshape([1,4])]

            # get sorted image matches
            # 6 is the lowes_distance_inverse, 7 is the heatmap value
            lowes_distances = matches_for_image[:, 6]
            heatmap_vals = matches_for_image[:, 7] / matches_for_image[:, 7].sum()
            score_list = lowes_distances * heatmap_vals # or you can use, score_list = lowes_distances

            # sorted_indices
            sorted_indices = np.argsort(score_list)
            # in descending order
            sorted_matches = matches_for_image[sorted_indices[::-1]]

            # prosac (or modified)
            start = time.time()
            inliers_no, ouliers_no, iterations, best_model = run_prosac(sorted_matches)
            end = time.time()
            elapsed_time = end - start

            modified_images_poses[image] = best_model
            modified_data = np.r_[modified_data, np.array([inliers_no, ouliers_no, iterations, elapsed_time]).reshape([1, 4])]
        else:
            print(image + " has less than 4 matches..")

    # NOTE: folders .../RANSAC_results/"+features_no+"/... where created manually..
    print("Saving Data..")
    np.save(vanillia_data_path_save_poses, vanilla_images_poses)
    np.save(vanillia_data_path_save_info, vanilla_data)
    np.save(modified_data_path_save_poses, modified_images_poses)
    np.save(modified_data_path_save_info, modified_data)

    print("\n")
    print("Results for exponential_decay_value " + str(exponential_decay_value/10) + ":")
    print("Vanillia")
    print("     Average Inliers: " + str(np.mean(vanilla_data[:,0])))
    print("     Average Outliers: " + str(np.mean(vanilla_data[:,1])))
    print("     Average Iterations: " + str(np.mean(vanilla_data[:,2])))
    print("     Average Time (s): " + str(np.mean(vanilla_data[:,3])))
    print("Modified")
    print("     Average Inliers: " + str(np.mean(modified_data[:, 0])))
    print("     Average Outliers: " + str(np.mean(modified_data[:, 1])))
    print("     Average Iterations: " + str(np.mean(modified_data[:, 2])))
    print("     Average Time (s): " + str(np.mean(modified_data[:, 3])))
    print("<---->")

    print("Done!")

# NOTE: for saving files, vanilla_ransac_images_pose and vanilla_ransac_data are repeating but it does not really matter
# because run_ransac_comparison and run_prosac_comparison will save the same data regarding those.

# colmap_features_no can be "2k", "1k", "0.5k", "0.25k"
# exponential_decay can be any of 0.1 to 0.9
features_no = "1k"
exponential_decay_value = 0.5

print("Running PROSAC comparison against un-weighted matches")
matches_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/"+features_no+"/matches_all.npy"
vanillia_data_path_save_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/vanilla_images_pose_"+str(exponential_decay_value)+".npy"
modified_data_path_save_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/modified_images_pose_"+str(exponential_decay_value)+".npy"

vanillia_data_path_save_info = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/vanillia_data_" + str(exponential_decay_value) + ".npy"
modified_data_path_save_info = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/modified_data_" + str(exponential_decay_value) + ".npy"

run_comparison("1k", 0.5, run_ransac, run_prosac, matches_path,
               vanillia_data_path_save_poses, modified_data_path_save_poses,
               vanillia_data_path_save_info, modified_data_path_save_info)



