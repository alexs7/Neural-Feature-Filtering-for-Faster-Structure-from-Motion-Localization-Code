import numpy as np
from ransac import run_ransac, run_ransac_modified
import time

# get the "sub_distributions" for each matches set for each image
def get_sub_distribution(matches_for_image, distribution):
    indices = matches_for_image[:, 5]
    indices = indices.astype(int)
    sub_distribution = distribution[0, indices]
    sub_distribution = sub_distribution / np.sum(sub_distribution)

    # This is a safety net for the "alt" version - TODO: elaborate
    if(np.nonzero(sub_distribution)[0].shape[0] < 4):
        return np.full((1, matches_for_image.shape[0]), 1 / matches_for_image.shape[0])

    return sub_distribution

# load localised images names
localised_images = []
path_to_query_images_file = '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/images_localised.txt'
with open(path_to_query_images_file) as f:
    localised_images = f.readlines()
localised_images = [x.strip() for x in localised_images]

# TODO: why not using matches_base ?!?!! and comparing to that ?
matches_all = np.load('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/matches_all.npy')

exponential_decay_values = np.linspace(0,1, num=10,endpoint=False)[1:10]

index_for_file_loading = 1 # 1 is for 0.1, 2 is for 0.2 etc etc (referring to exponential decay)
for exponential_decay_value in exponential_decay_values:

    #ordinary distributions, and altered distributions (the one that has values over the mean), both same size as 3D points
    ord_distribution = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/heatmap_matrix_avg_points_values_" + str(index_for_file_loading) + ".txt")
    ord_distribution = ord_distribution.reshape([1, ord_distribution.shape[0]])

    alt_distribution = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/heatmap_matrix_avg_points_values_" + str(index_for_file_loading) + "_mean_distri.txt")
    alt_distribution = alt_distribution.reshape([1, alt_distribution.shape[0]])

    print("Getting sub_distributions..")
    ord_sub_distributions = {}
    alt_sub_distributions = {}
    for image in localised_images:
        matches_for_image = matches_all.item()[image]
        ord_sub_distributions[image] = get_sub_distribution(matches_for_image, ord_distribution)
        alt_sub_distributions[image] = get_sub_distribution(matches_for_image, alt_distribution)

    print("Running RANSAC versions.. for exponential decay of value: " + str(exponential_decay_value))

    #  this will hold inliers_no, ouliers_no, iterations, time for each image
    vanilla_ransac_data = np.empty([0, 4])
    vanilla_ransac_images_poses = {}

    ord_modified_ransac_data = np.empty([0, 4])
    ord_modified_ransac_images_poses = {}

    alt_modified_ransac_data = np.empty([0, 4])
    alt_modified_ransac_images_poses = {}

    for image in localised_images:
        matches_for_image = matches_all.item()[image]

        if(len(matches_for_image) >= 4):
            # print("Doing image: " + image + ", with no of matches: " + str(len(matches_for_image)))
            # vanilla
            start = time.time()
            inliers_no, ouliers_no, iterations, best_model, elapsed_time_total_for_random_sampling = run_ransac(matches_for_image)
            end  = time.time()
            elapsed_time = end - start - elapsed_time_total_for_random_sampling

            vanilla_ransac_images_poses[image] = best_model
            vanilla_ransac_data = np.r_[vanilla_ransac_data, np.array([inliers_no, ouliers_no, iterations, elapsed_time]).reshape([1,4])]

            # my version using VM ord distris
            start = time.time()
            inliers_no, ouliers_no, iterations, best_model, elapsed_time_total_for_random_sampling = run_ransac_modified(matches_for_image, ord_sub_distributions[image])
            end = time.time()
            elapsed_time = end - start - elapsed_time_total_for_random_sampling

            ord_modified_ransac_images_poses[image] = best_model
            ord_modified_ransac_data = np.r_[ord_modified_ransac_data, np.array([inliers_no, ouliers_no, iterations, elapsed_time]).reshape([1, 4])]

            # my version using VM alt distris TODO: fix
            # start = time.time()
            # inliers_no, ouliers_no, iterations, best_model, elapsed_time_total_for_random_sampling = run_ransac_modified(matches_for_image, alt_sub_distributions[image])
            # end = time.time()
            # elapsed_time = end - start - elapsed_time_total_for_random_sampling
            #
            # alt_modified_ransac_images_poses[image] = best_model
            # alt_modified_ransac_data = np.r_[alt_modified_ransac_data, np.array([inliers_no, ouliers_no, iterations, elapsed_time]).reshape([1, 4])]
        else:
            print(image + " has less than 4 matches..")

    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/vanilla_ransac_images_pose_" + str(index_for_file_loading) + ".npy", vanilla_ransac_images_poses)
    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/vanilla_ransac_data_" + str(index_for_file_loading) + ".npy", vanilla_ransac_data)

    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/modified_ransac_images_pose_ord_" + str(index_for_file_loading) + ".npy", ord_modified_ransac_images_poses)
    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/modified_ransac_data_ord_" + str(index_for_file_loading) + ".npy", ord_modified_ransac_data)

    # np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/modified_ransac_images_pose_alt_" + str(index_for_file_loading) + ".npy", alt_modified_ransac_images_poses)
    # np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/modified_ransac_data_alt_" + str(index_for_file_loading) + ".npy", alt_modified_ransac_data)

    print("Results for exponential_decay_value " + str(index_for_file_loading/10) + ":")
    print("Vanillia RANSAC")
    print("     Average Inliers: " + str(np.mean(vanilla_ransac_data[:,0])))
    print("     Average Outliers: " + str(np.mean(vanilla_ransac_data[:,1])))
    print("     Average Time (s): " + str(np.mean(vanilla_ransac_data[:,3])))
    print("     Average Iterations: " + str(np.mean(vanilla_ransac_data[:,2])))
    print("Modified RANSAC with ord_distributions")
    print("     Average Inliers: " + str(np.mean(ord_modified_ransac_data[:, 0])))
    print("     Average Outliers: " + str(np.mean(ord_modified_ransac_data[:, 1])))
    print("     Average Time (s): " + str(np.mean(ord_modified_ransac_data[:, 3])))
    print("     Average Iterations: " + str(np.mean(ord_modified_ransac_data[:, 2])))
    print("Modified RANSAC with alt_distributions")
    print("     Average Inliers: " + str(np.mean(alt_modified_ransac_data[:, 0])))
    print("     Average Outliers: " + str(np.mean(alt_modified_ransac_data[:, 1])))
    print("     Average Time (s): " + str(np.mean(alt_modified_ransac_data[:, 3])))
    print("     Average Iterations: " + str(np.mean(alt_modified_ransac_data[:, 2])))

    index_for_file_loading = index_for_file_loading + 1

    print("<---->")
    breakpoint()

print("Done!")




