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
    lowes_distances = matches[:, 6]
    heatmap_vals = matches[:, 7]
    score_list = lowes_distances * heatmap_vals
    # sorted_indices
    sorted_indices = np.argsort(score_list)
    # in descending order
    sorted_matches = matches[sorted_indices[::-1]]
    return sorted_matches

def enhanced_sort_matches_kmeans(matches):
    xs = []
    ys = []
    sorting_vals = []
    for i in range(len(matches)):
        x = matches[i, 0]
        y = matches[i, 1]
        val = matches[i, 6] * matches[i, 7] #lowes_distances * heatmap_vals
        xs.append(x)
        ys.append(y)
        sorting_vals.append(val)
    xs = np.array(xs)
    xs = xs.reshape([xs.shape[0], 1])
    ys = np.array(ys)
    ys = ys.reshape([ys.shape[0], 1])
    sorting_vals = np.array(sorting_vals)
    sorting_vals = sorting_vals.reshape([sorting_vals.shape[0], 1])
    data = np.concatenate([xs, ys, sorting_vals], axis=1)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(data)

    # matches indices of centroids
    idx0 = np.argmin(kmeans.transform(data)[:, 0])
    idx1 = np.argmin(kmeans.transform(data)[:, 1])
    idx2 = np.argmin(kmeans.transform(data)[:, 2])
    idx3 = np.argmin(kmeans.transform(data)[:, 3])

    top_idx = [idx0, idx1, idx2, idx3]
    remaining_matches = np.delete(matches, top_idx, axis=0)
    lowes_distances = remaining_matches[:, 6]
    heatmap_vals = remaining_matches[:, 7]
    score_list = lowes_distances * heatmap_vals
    # sorted_indices
    sorted_indices = np.argsort(score_list)
    # in descending order
    remaining_sorted_matches = remaining_matches[sorted_indices[::-1]]
    top_matches = matches[top_idx,:]
    sorted_matches = np.r_[top_matches, remaining_sorted_matches]

    return sorted_matches

def plain_sort_matches(matches):
    lowes_distances = matches[:, 6]
    score_list = lowes_distances
    # sorted_indices
    sorted_indices = np.argsort(score_list)
    # in descending order
    sorted_matches = matches[sorted_indices[::-1]]
    return sorted_matches

def run_comparison(func, matches_path, test_images, points3D_avg_heatmap_vals = np.array([]) , sort_matches_func = None):

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

            if(sort_matches_func !=None):
                matches_for_image = sort_matches_func(matches_for_image)

            if(points3D_avg_heatmap_vals.size !=0 ):
                sub_dist = get_sub_distribution(matches_for_image, points3D_avg_heatmap_vals)
                matches_for_image = np.hstack((matches_for_image, sub_dist))

            start = time.time()
            inliers_no, outliers_no, iterations, best_model, inliers = func(matches_for_image)
            end  = time.time()
            elapsed_time = end - start

            images_poses[image] = best_model
            data = np.r_[data, np.array([inliers_no, outliers_no, iterations, elapsed_time]).reshape([1,4])]
        else:
            print(image + " has less than 4 matches..")
    print("\n")

    return images_poses, data
