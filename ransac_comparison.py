import numpy as np
from sklearn.cluster import KMeans

from query_image import load_images_from_text_file, read_images_binary, get_query_image_global_pose_new_model
import time

# This is used with the old modified RANSAC version
# get the "sub_distributions" for each matches set for each image - This will have to be relooked at!
def get_sub_distribution(matches_for_image, distribution):
    indices = matches_for_image[:, 5]
    indices = indices.astype(int)
    sub_distribution = distribution[0, indices]
    sub_distribution = sub_distribution / np.sum(sub_distribution)
    sub_distribution = sub_distribution.reshape([sub_distribution.shape[0], 1])
    return sub_distribution

# functions for PROSAC's score list
def enhanced_sort_matches(matches):
    lowes_distances = matches[:, 6] / matches[:,6].sum()
    scores = matches[:, 7] / matches[:,7].sum()
    score_list = lowes_distances * scores
    # sorted_indices
    sorted_indices = np.argsort(score_list)
    # in descending order
    sorted_matches = matches[sorted_indices[::-1]]
    return sorted_matches

def sort_matches(matches, idx):
    score_list = matches[:, idx]
    # sorted_indices
    sorted_indices = np.argsort(score_list)
    # in descending order
    sorted_matches = matches[sorted_indices[::-1]]
    return sorted_matches

def run_comparison(func, matches_path, test_images, dist_vals = np.array([]), sort_matches_func = None, val_idx = None):

    # Question: why not using matches_base ?!?!! and comparing to that ?
    # Answer: and compare what ? ransac will take less time with matches_base given the smaller number of all matches,
    # plus the get_sub_distribution might not make sense here as for base there are no future sessions yet.
    matches_all = np.load(matches_path)

    #  this will hold inliers_no, outliers_no, iterations, time for each image
    data = np.empty([0, 4])
    images_poses = {}

    for i in range(len(test_images)):
        image = test_images[i]
        matches_for_image = matches_all.item()[image]
        print("Doing image " + str(i+1) + "/" + str(len(test_images)) + ", " + image , end="\r")

        if(len(matches_for_image) >= 4):

            if(sort_matches_func !=None and val_idx != None):
                matches_for_image = sort_matches_func(matches_for_image,val_idx)

            if(dist_vals.size !=0):
                dist_vals = dist_vals / dist_vals.sum()
                sub_dist = get_sub_distribution(matches_for_image, dist_vals)
                matches_for_image = np.hstack((matches_for_image, sub_dist))

            start = time.time()
            inliers_no, outliers_no, iterations, best_model, inliers = func(matches_for_image,image)
            end  = time.time()
            elapsed_time = end - start

            images_poses[image] = best_model
            data = np.r_[data, np.array([inliers_no, outliers_no, iterations, elapsed_time]).reshape([1,4])]
        else:
            print(image + " has less than 4 matches..")
    print("\n")

    return images_poses, data
