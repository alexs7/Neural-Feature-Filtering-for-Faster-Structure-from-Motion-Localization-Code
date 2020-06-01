import numpy as np

from query_image import load_images_from_text_file
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

def run_ransac_comparison(features_no):

    print("-- Doing features_no " + features_no + " --")

    # load localised images names (including base ones)
    localised_query_images = load_images_from_text_file("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/images_localised_and_not_localised/" + features_no + "/images_localised.txt")
    # of course base images will be localised..
    base_images = load_images_from_text_file("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/base_images.txt")

    # TODO: why not using matches_base ?!?!! and comparing to that ?
    # and compare what ? ransac will take less time with matches_base given the smaller number of matches, plus the get_sub_distribution might not make sense
    # here as for base there are no future sessions yet.
    matches_all = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/"+features_no+"/matches_all.npy")

    for exponential_decay_value in range(1,10):

        #ordinary distributions, and altered distributions (the one that has values over the mean), both same size as 3D points
        points3D_avg_heatmap_vals = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/"+features_no+"/heatmap_matrix_avg_points_values_" + str(exponential_decay_value) + ".txt")
        points3D_avg_heatmap_vals = points3D_avg_heatmap_vals.reshape([1, points3D_avg_heatmap_vals.shape[0]])

        print("Getting sub_distributions..")
        ord_sub_distributions = {}
        for image in localised_query_images:
            if(image not in base_images):
                matches_for_image = matches_all.item()[image]
                ord_sub_distributions[image] = get_sub_distribution(matches_for_image, points3D_avg_heatmap_vals)

        print("Running RANSAC versions.. for exponential decay of value: " + str(exponential_decay_value/10))

        #  this will hold inliers_no, ouliers_no, iterations, time for each image
        vanilla_ransac_data = np.empty([0, 4])
        vanilla_ransac_images_poses = {}

        ord_modified_ransac_data = np.empty([0, 4])
        ord_modified_ransac_images_poses = {}

        alt_modified_ransac_data = np.empty([0, 4])

        image_index = 1
        for image in localised_query_images:
            if (image not in base_images):
                matches_for_image = matches_all.item()[image]
                print("Doing image " + str(image_index) + "/" + str(len(localised_query_images) - len(base_images)) + ", " + image , end="\r")

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
                else:
                    print(image + " has less than 4 matches..")

                image_index = image_index + 1

        # NOTE: folders .../RANSAC_results/"+features_no+"/... where created manually..
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/vanilla_ransac_images_pose_" + str(exponential_decay_value) + ".npy", vanilla_ransac_images_poses)
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/vanilla_ransac_data_" + str(exponential_decay_value) + ".npy", vanilla_ransac_data)

        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/modified_ransac_images_pose_" + str(exponential_decay_value) + ".npy", ord_modified_ransac_images_poses)
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/modified_ransac_data_" + str(exponential_decay_value) + ".npy", ord_modified_ransac_data)

        print("\n")
        print("Results for exponential_decay_value " + str(exponential_decay_value/10) + ":")
        print("Vanillia RANSAC")
        print("     Average Inliers: " + str(np.mean(vanilla_ransac_data[:,0])))
        print("     Average Outliers: " + str(np.mean(vanilla_ransac_data[:,1])))
        print("     Average Iterations: " + str(np.mean(vanilla_ransac_data[:,2])))
        print("     Average Time (s): " + str(np.mean(vanilla_ransac_data[:,3])))
        print("Modified RANSAC with ord_distributions")
        print("     Average Inliers: " + str(np.mean(ord_modified_ransac_data[:, 0])))
        print("     Average Outliers: " + str(np.mean(ord_modified_ransac_data[:, 1])))
        print("     Average Iterations: " + str(np.mean(ord_modified_ransac_data[:, 2])))
        print("     Average Time (s): " + str(np.mean(ord_modified_ransac_data[:, 3])))
        print("<---->")


    print("Done!")

colmap_features_no = ["2k", "1k", "0.5k", "0.25k"]
# run for each no of features
for features_no in colmap_features_no:
    run_ransac_comparison(features_no)

