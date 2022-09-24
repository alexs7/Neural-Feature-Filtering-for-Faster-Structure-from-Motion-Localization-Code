# 15/09/2022 - major refactor
import numpy as np
import time
from tqdm import tqdm

from RANSACParameters import RANSACParameters

# Example:  match_data = [[x, y, x, y, z , m.distance, n.distance], [h_m, h_n, r_m, r_n, v_m, v_n]] -> but flatten
# [[x (0), y (1), x (2), y (3), z (4), m.distance (5), n.distance (6)], [h_m (7), h_n (8), r_m (9), r_n (10), v_m (11), v_n (12)]]
# first value is of m (the closest match), second value is of n (second closest).
# h = heatmap
# r = reliability
# v = visibility

def get_sub_distribution(matches_for_image, index):
    vals = matches_for_image[:, index]
    vals[vals < 0] = 0 # some values from NN might be negative
    sub_distribution = vals / np.sum(vals)
    sub_distribution = sub_distribution.reshape([sub_distribution.shape[0], 1])
    return sub_distribution

# lowes_distance_inverse = n.distance / m.distance  # inverse here as the higher the better for PROSAC
def lowes_distance_inverse(matches):
    return matches[:, 6] / matches[:, 5]

def heatmap_val(matches):
    return matches[:, 7]

def reliability_score(matches):
    return matches[:, 9]

def reliability_score_ratio(matches):
    return np.nan_to_num(matches[:, 9] / matches[:, 10], nan = 0.0, neginf = 0.0, posinf = 0.0)

def heatmap_value_ratio(matches):
    return matches[:, 7] / matches[:, 8]

def lowes_ratio_by_higher_reliability_score(matches):
    scores = []
    for match in matches:
        lowes_distance_inverse = match[6] / match[5]
        score_m = match[9]
        score_n = match[10]
        higher_score = score_m if score_m > score_n else score_n
        final_score = lowes_distance_inverse * higher_score
        scores.append(final_score)
    return np.array(scores)

def lowes_ratio_by_higher_heatmap_val(matches):
    values = []
    for match in matches:
        lowes_distance_inverse = match[6] / match[5]
        val_m = match[7]
        val_n = match[8]
        higher_val = val_m if val_m > val_n else val_n
        final_score = lowes_distance_inverse * higher_val
        values.append(final_score)
    return np.array(values)

def lowes_ratio_reliability_score_ratio(matches):
    scores = []
    for match in matches:
        lowes_distance_inverse = match[6] / match[5]
        reliability_score_ratio = match[9] / match[10]
        final_score = lowes_distance_inverse * reliability_score_ratio
        scores.append(final_score)
    return np.array(scores)

def lowes_ratio_heatmap_val_ratio(matches):
    scores = []
    for match in matches:
        lowes_distance_inverse = match[6] / match[5]
        heatmap_val_ratio =  match[7] /  match[8]
        final_score = lowes_distance_inverse * heatmap_val_ratio
        scores.append(final_score)
    return np.array(scores)

def higher_neighbour_score(matches):
    scores = []
    for match in matches:
        score_m = match[9]
        score_n = match[10]
        higher_score = score_m if score_m > score_n else score_n
        scores.append(higher_score)
    return np.array(scores)

def higher_neighbour_value(matches):
    values = []
    for match in matches:
        value_m = match[7]
        value_n = match[8]
        higher_value = value_m if value_m > value_n else value_n
        values.append(higher_value)
    return np.array(values)

def higher_neighbour_visibility_score(matches):
    scores = []
    for match in matches:
        score_m = match[11]
        score_n = match[12]
        higher_score = score_m if score_m > score_n else score_n
        scores.append(higher_score)
    return np.array(scores)

functions = {RANSACParameters.lowes_distance_inverse_ratio_index : lowes_distance_inverse,
             RANSACParameters.heatmap_val_index : heatmap_val,
             RANSACParameters.reliability_score_index : reliability_score,
             RANSACParameters.reliability_score_ratio_index : reliability_score_ratio,
             RANSACParameters.lowes_ratio_reliability_score_val_ratio_index : lowes_ratio_reliability_score_ratio,
             RANSACParameters.lowes_ratio_heatmap_val_ratio_index : lowes_ratio_heatmap_val_ratio,
             RANSACParameters.higher_neighbour_score_index : higher_neighbour_score,
             RANSACParameters.heatmap_val_ratio_index: heatmap_value_ratio,
             RANSACParameters.higher_neighbour_val_index: higher_neighbour_value,
             RANSACParameters.higher_visibility_score_index: higher_neighbour_visibility_score,
             RANSACParameters.lowes_ratio_by_higher_reliability_score_index: lowes_ratio_by_higher_reliability_score,
             RANSACParameters.lowes_ratio_by_higher_heatmap_val_index: lowes_ratio_by_higher_heatmap_val}

def sort_matches(matches, idx):
    score_list = functions[idx](matches)
    # sorted_indices
    sorted_indices = np.argsort(score_list)
    # in descending order ([::-1] makes it from ascending to descending )
    sorted_matches = matches[sorted_indices[::-1]]
    return sorted_matches

# 01/09/2022, This is used to return data per image! So I can examine later one by one!
def run_comparison(func, matches, test_images, intrinsics, val_idx = None):

    #  this will hold inliers_no, outliers_no, iterations, time for each image
    images_data = {}

    for i in tqdm(range(len(test_images))):
        image = test_images[i]
        matches_for_image = matches[image]

        if (len(matches_for_image) < 4): #did not get a pose move to next image
            # est_pose, inliers_no, outliers_no, iterations, elapsed_time
            images_data[image] = [None, None, None, None, None]
            continue

        if(val_idx is not None): #When using prosac the matches are already sorted so val_idx is None
            if(val_idx < 0):
                sub_dist = get_sub_distribution(matches_for_image, 7)
                matches_for_image = np.hstack((matches_for_image, sub_dist))

        start = time.time()
        best_model = func(matches_for_image, intrinsics)

        if(best_model == None): #degenerate case
            images_data[image] = [None, None, None, None, None]
            continue

        end = time.time()
        elapsed_time = end - start

        est_pose = best_model['Rt']
        inliers_no = best_model['inliers_no']
        outliers_no = best_model['outliers_no']
        iterations = best_model['iterations']

        images_data[image] = [est_pose, inliers_no, outliers_no, iterations, elapsed_time]

    return images_data