import numpy as np
from query_image import load_images_from_text_file, read_images_binary, get_query_image_global_pose_new_model
import time

# This was used with the old modified RANSAC version
# get the "sub_distributions" for each matches set for each image - This will have to be relooked at!
# def get_sub_distribution(matches_for_image, distribution):
#     indices = matches_for_image[:, 5]
#     indices = indices.astype(int)
#     sub_distribution = distribution[0, indices]
#     sub_distribution = sub_distribution / np.sum(sub_distribution)
#     return sub_distribution

def run_comparison(exponential_decay_value, ransac, prosac, matches_path, test_images):

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

    for i in range(len(test_images)):
        image = test_images[i]
        matches_for_image = matches_all.item()[image]
        print("Doing image " + str(i+1) + "/" + str(len(test_images)) + ", " + image , end="\r")

        if(len(matches_for_image) >= 4):
            # vanilla
            start = time.time()
            inliers_no, ouliers_no, iterations, best_model, inliers = ransac(matches_for_image)
            end  = time.time()
            elapsed_time = end - start

            vanilla_images_poses[image] = best_model
            vanilla_data = np.r_[vanilla_data, np.array([inliers_no, ouliers_no, iterations, elapsed_time]).reshape([1,4])]

            # get sorted image matches
            # 6 is the lowes_distance_inverse, 7 is the heatmap value
            # TODO: Normalise both ?
            lowes_distances = matches_for_image[:, 6]
            heatmap_vals = matches_for_image[:, 7] / matches_for_image[:, 7].sum()
            score_list = lowes_distances * heatmap_vals # or you can use, score_list = lowes_distances

            # sorted_indices
            sorted_indices = np.argsort(score_list)
            # in descending order
            sorted_matches = matches_for_image[sorted_indices[::-1]]

            # prosac (or modified)
            start = time.time()
            inliers_no_mod, ouliers_no_mod, iterations_mod, best_model_mod, inliers_mod = prosac(sorted_matches)
            end = time.time()
            elapsed_time_mod = end - start

            modified_images_poses[image] = best_model_mod
            modified_data = np.r_[modified_data, np.array([inliers_no_mod, ouliers_no_mod, iterations_mod, elapsed_time_mod]).reshape([1, 4])]
        else:
            print(image + " has less than 4 matches..")

    print("\n")
    print("Results for exponential_decay_value " + str(exponential_decay_value) + ":")
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

    # NOTE: for saving files, vanilla_ransac_images_pose and vanilla_ransac_data are repeating but it does not really matter
    # because run_ransac_comparison and run_prosac_comparison will save the same data regarding those.
    return vanilla_images_poses, vanilla_data, modified_images_poses, modified_data
