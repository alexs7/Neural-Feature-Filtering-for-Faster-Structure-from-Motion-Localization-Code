import numpy as np
import time

# Note:  match_data = [[x, y, x, y, z , m.distance, n.distance], scores]
# where scores are k by 2 arrays, same size as many points3D scores you pass (in my case 2,heatmap_matrix_avg_points_values, reliability_scores)
# first value is of m (the closest match), second value is of n (second closest).
from parameters import Parameters


def get_sub_distribution(matches_for_image):
    vals = matches_for_image[:, 6] #TODO: this can change to heatmap value, I am using reliability score because it might give better results
    sub_distribution = vals / np.sum(vals)
    sub_distribution = sub_distribution.reshape([sub_distribution.shape[0], 1])
    return sub_distribution

# prosac different scores to sort by - can try more
# some are below:
# score_ratio = score_m / score_n  # the higher the better (first match is more "static" than the second, ratio)
# custom_score = lowes_distance_inverse * score_ratio  # self-explanatory
# from the 2 neighbours reliability scores, choose the higher one
# higher_neighbour_score = score_m if score_m > score_n else score_n

# lowes_distance_inverse = n.distance / m.distance  # inverse here as the higher the better for PROSAC
def lowes_distance_inverse(matches):
    return matches[:, 6] / matches[:, 5]

def heatmap_val(matches):
    return matches[:, 7]

def reliability_score(matches):
    return matches[:, 9]

def reliability_score_ratio(matches):
    return matches[:, 9] / matches[:, 10]

def custom_score(matches):
    lowes_distance_inverse = matches[:, 6] / matches[:, 5]
    score_ratio = matches[:, 9] / matches[:, 10]
    return lowes_distance_inverse * score_ratio

def higher_neighbour_score(matches):
    scores = []
    for match in matches:
        score_m = match[9]
        score_n = match[10]
        higher_score = score_m if score_m > score_n else score_n
        scores.append(higher_score)
    return np.array(scores)

functions = {Parameters.lowes_distance_inverse_ratio_index : lowes_distance_inverse,
             Parameters.heatmap_val_index : heatmap_val,
             Parameters.reliability_score_index : reliability_score,
             Parameters.reliability_score_ratio_index : reliability_score_ratio,
             Parameters.custom_score_index : custom_score,
             Parameters.higher_neighbour_score_index : higher_neighbour_score}

def sort_matches(matches, idx):
    score_list = functions[idx](matches)
    # sorted_indices
    sorted_indices = np.argsort(score_list)
    # in descending order ([::-1] makes it from ascending to descending )
    sorted_matches = matches[sorted_indices[::-1]]
    return sorted_matches

def run_comparison(func, matches, test_images, verbose = True, val_idx = None):

    #  this will hold inliers_no, outliers_no, iterations, time for each image
    data = np.empty([0, 4])
    images_poses = {}

    for i in range(len(test_images)):
        image = test_images[i]
        matches_for_image = matches[image]
        # print("Doing image " + str(i+1) + "/" + str(len(test_images)) + ", " + image , end="\r")

        assert(len(matches_for_image) >= 4)

        if(val_idx is not None):
            if(val_idx != -1):
                matches_for_image = sort_matches(matches_for_image, val_idx)

            if(val_idx == -1):
                sub_dist = get_sub_distribution(matches_for_image)
                matches_for_image = np.hstack((matches_for_image, sub_dist))

        start = time.time()
        best_model = func(matches_for_image)
        end  = time.time()
        elapsed_time = end - start

        pose = best_model['Rt']
        inliers_no = best_model['inliers_no']
        outliers_no = best_model['outliers_no']
        iterations = best_model['iterations']

        images_poses[image] = pose
        data = np.r_[data, np.array([inliers_no, outliers_no, iterations, elapsed_time]).reshape([1,4])]

    if(verbose):
        print("[inliers_no, outliers_no, iterations, elapsed_time]:")
        print(data.mean(axis=0))

    return images_poses, data
