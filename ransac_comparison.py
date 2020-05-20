import numpy as np
from ransac import run_ransac, run_ransac_modified
import time

# load localised images names
localised_images = []
path_to_query_images_file = '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/images_localised.txt'
with open(path_to_query_images_file) as f:
    localised_images = f.readlines()
localised_images = [x.strip() for x in localised_images]

matches_all = np.load('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/matches_all.npy')

print("Running Vanillia RANSAC")
# run vanillia
#  this will hold inliers_no, ouliers_no, iterations, time for each image
vanilla_ransac_data = np.empty([0, 4])
vanilla_ransac_images_pose = {}
for image in localised_images:
    matches_for_image = matches_all.item()[image]

    start = time.time()
    inliers_no, ouliers_no, iterations, best_model = run_ransac(matches_for_image)
    end  = time.time()
    elapsed_time = end - start

    vanilla_ransac_images_pose[image] = best_model
    vanilla_ransac_data = np.r_[vanilla_ransac_data, np.array([inliers_no, ouliers_no, iterations, elapsed_time]).reshape([1,4])]

np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/vanilla_ransac_images_pose.npy", vanilla_ransac_images_pose)
np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/vanilla_ransac_data.npy", vanilla_ransac_data)

# run my version
distribution = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/heatmap_matrix_avg_points_values.txt")
distribution = distribution.reshape([1, distribution.shape[0]])

heatmap_matrix = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/heatmap_matrix.txt")

print("Getting the sub_distribution for matches")
# get the "sub_distributions" for each matches set for each image
def get_sub_distribution(matches_for_image, distribution):
    indices = matches_for_image[:, 5]
    indices = indices.astype(int)
    sub_distribution = distribution[0, indices]
    sub_distribution = sub_distribution / np.sum(sub_distribution)
    return sub_distribution

sub_distributions = {}
for image in localised_images:
    matches_for_image = matches_all.item()[image]
    sub_distributions[image] = get_sub_distribution(matches_for_image, distribution)

print("Running Modified RANSAC")
modified_ransac_data = np.empty([0, 4])
modified_ransac_images_pose = {}
for image in localised_images:
    matches_for_image = matches_all.item()[image]

    start = time.time()
    inliers_no, ouliers_no, iterations, best_model = run_ransac_modified(matches_for_image, sub_distributions[image])
    end  = time.time()
    elapsed_time = end - start

    modified_ransac_images_pose[image] = best_model
    modified_ransac_data = np.r_[modified_ransac_data, np.array([inliers_no, ouliers_no, iterations, elapsed_time]).reshape([1,4])]

np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/modified_ransac_images_pose.npy", modified_ransac_images_pose)
np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/modified_ransac_data.npy", modified_ransac_data)




