import numpy as np
from query_image import load_images_from_text_file, read_images_binary, get_query_image_global_pose_new_model
import time

# This is used with the old modified RANSAC version
# get the "sub_distributions" for each matches set for each image - This will have to be relooked at!
def get_sub_distribution(matches_for_image, distribution):
    indices = matches_for_image[:, 5]
    indices = indices.astype(int)
    sub_distribution = distribution[0, indices]
    sub_distribution = sub_distribution / np.sum(sub_distribution)
    return sub_distribution

def run_comparison(exponential_decay_value, ransac, run_ransac_modified, prosac, matches_path, test_images, points3D_avg_heatmap_vals):

    # Question: why not using matches_base ?!?!! and comparing to that ?
    # Answer: and compare what ? ransac will take less time with matches_base given the smaller number of all matches,
    # plus the get_sub_distribution might not make sense here as for base there are no future sessions yet.
    matches_all = np.load(matches_path)

    #  this will hold inliers_no, outliers_no, iterations, time for each image
    ransac_data = np.empty([0, 4])
    ransac_images_poses = {}

    ransac_dist_data = np.empty([0, 4])
    ransac_dist_images_poses = {}

    prosac_data = np.empty([0, 4])
    prosac_images_poses = {}

    for i in range(len(test_images)):
        image = test_images[i]
        matches_for_image = matches_all.item()[image]
        print("Doing image " + str(i+1) + "/" + str(len(test_images)) + ", " + image , end="\r")

        if(len(matches_for_image) >= 4):
            # vanilla RANSAC
            start = time.time()
            inliers_no, outliers_no, iterations, best_model, inliers = ransac(matches_for_image)
            end  = time.time()
            elapsed_time = end - start

            ransac_images_poses[image] = best_model
            ransac_data = np.r_[ransac_data, np.array([inliers_no, outliers_no, iterations, elapsed_time]).reshape([1,4])]

            # modified RANSAC with distribution
            sub_dist = get_sub_distribution(matches_for_image, points3D_avg_heatmap_vals)
            start = time.time()
            inliers_no, outliers_no, iterations, best_model, inliers = run_ransac_modified(matches_for_image,sub_dist)
            end = time.time()
            elapsed_time = end - start

            ransac_dist_images_poses[image] = best_model
            ransac_dist_data = np.r_[ransac_dist_data, np.array([inliers_no, outliers_no, iterations, elapsed_time]).reshape([1, 4])]

            # PROSAC
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

            start = time.time()
            inliers_no, outliers_no, iterations, best_model, inliers = prosac(sorted_matches)
            end = time.time()
            elapsed_time_mod = end - start

            prosac_images_poses[image] = best_model
            prosac_data = np.r_[prosac_data, np.array([inliers_no, outliers_no, iterations, elapsed_time_mod]).reshape([1, 4])]

        else:
            print(image + " has less than 4 matches..")

    print("\n")
    print("Results for exponential_decay_value " + str(exponential_decay_value) + ":")
    print("Vanillia RANSAC")
    print("     Average Inliers: " + str(np.mean(ransac_data[:,0])))
    print("     Average Outliers: " + str(np.mean(ransac_data[:,1])))
    print("     Average Iterations: " + str(np.mean(ransac_data[:,2])))
    print("     Average Time (s): " + str(np.mean(ransac_data[:,3])))
    print("RANSAC with dist")
    print("     Average Inliers: " + str(np.mean(ransac_dist_data[:, 0])))
    print("     Average Outliers: " + str(np.mean(ransac_dist_data[:, 1])))
    print("     Average Iterations: " + str(np.mean(ransac_dist_data[:, 2])))
    print("     Average Time (s): " + str(np.mean(ransac_dist_data[:, 3])))
    print("PROSAC")
    print("     Average Inliers: " + str(np.mean(prosac_data[:, 0])))
    print("     Average Outliers: " + str(np.mean(prosac_data[:, 1])))
    print("     Average Iterations: " + str(np.mean(prosac_data[:, 2])))
    print("     Average Time (s): " + str(np.mean(prosac_data[:, 3])))
    print("<---->")

    return ransac_images_poses, ransac_data, ransac_dist_images_poses, ransac_dist_data, prosac_images_poses, prosac_data
