import numpy as np

from point3D_loader import read_points3d_default
from query_image import load_images_from_text_file, read_images_binary
from ransac import run_ransac, run_ransac_modified, prosac
import time

# get the "sub_distributions" for each matches set for each image
def get_sub_distribution(matches_for_image, distribution):
    indices = matches_for_image[:, 5]
    indices = indices.astype(int)
    sub_distribution = distribution[0, indices]
    sub_distribution = sub_distribution / np.sum(sub_distribution)
    return sub_distribution

def run_ransac_comparison(features_no, exponential_decay_value, weighted=False):

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
    if(weighted): #if statement here is self explanatory
        matches_all = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/"+features_no+"/matches_all_weighted.npy")
    else:
        matches_all = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/" + features_no + "/matches_all.npy")

    #distribution; row vector, same size as 3D points
    points3D_avg_heatmap_vals = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/"+features_no+"/heatmap_matrix_avg_points_values_" + str(exponential_decay_value) + ".txt")
    points3D_avg_heatmap_vals = points3D_avg_heatmap_vals.reshape([1, points3D_avg_heatmap_vals.shape[0]])

    print("Getting sub_distributions..")
    distributions = {}
    for image in localised_query_images_only:
        matches_for_image = matches_all.item()[image]
        distributions[image] = get_sub_distribution(matches_for_image, points3D_avg_heatmap_vals)

    print("Running RANSAC versions.. for exponential decay of value: " + str(exponential_decay_value))

    #  this will hold inliers_no, ouliers_no, iterations, time for each image
    vanilla_ransac_data = np.empty([0, 4])
    vanilla_ransac_images_poses = {}

    ord_modified_ransac_data = np.empty([0, 4])
    ord_modified_ransac_images_poses = {}

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

            vanilla_ransac_images_poses[image] = best_model
            vanilla_ransac_data = np.r_[vanilla_ransac_data, np.array([inliers_no, ouliers_no, iterations, elapsed_time]).reshape([1,4])]

            # my version using VM ord distris
            start = time.time()
            inliers_no, ouliers_no, iterations, best_model = run_ransac_modified(matches_for_image, distributions[image])
            end = time.time()
            elapsed_time = end - start

            ord_modified_ransac_images_poses[image] = best_model
            ord_modified_ransac_data = np.r_[ord_modified_ransac_data, np.array([inliers_no, ouliers_no, iterations, elapsed_time]).reshape([1, 4])]
        else:
            print(image + " has less than 4 matches..")

    # NOTE: folders .../RANSAC_results/"+features_no+"/... where created manually..
    if(weighted):
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/vanilla_ransac_images_pose_" + str(exponential_decay_value) + "_weighted.npy", vanilla_ransac_images_poses)
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/vanilla_ransac_data_" + str(exponential_decay_value) + "_weighted.npy", vanilla_ransac_data)

        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/modified_ransac_images_pose_" + str(exponential_decay_value) + "_weighted.npy", ord_modified_ransac_images_poses)
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/modified_ransac_data_" + str(exponential_decay_value) + "_weighted.npy", ord_modified_ransac_data)

        print("\n")
        print("Weighted Matches Results for exponential_decay_value " + str(exponential_decay_value/10) + ":")
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
    else:
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/vanilla_ransac_images_pose_" + str(exponential_decay_value) + ".npy", vanilla_ransac_images_poses)
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/vanilla_ransac_data_" + str(exponential_decay_value) + ".npy", vanilla_ransac_data)

        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/modified_ransac_images_pose_" + str(exponential_decay_value) + ".npy", ord_modified_ransac_images_poses)
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/modified_ransac_data_" + str(exponential_decay_value) + ".npy", ord_modified_ransac_data)

        print("\n")
        print("Non-Weighted Matches Results for exponential_decay_value " + str(exponential_decay_value/10) + ":")
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

def run_prosac_comparison(features_no, exponential_decay_value, weighted=False):

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
    if(weighted): #if statement here is self explanatory
        matches_all = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/"+features_no+"/matches_all_weighted.npy")
    else:
        matches_all = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/" + features_no + "/matches_all.npy")

    print("Running RANSAC/PROSAC.. for exponential decay of value: " + str(exponential_decay_value))

    #  this will hold inliers_no, ouliers_no, iterations, time for each image
    vanilla_ransac_data = np.empty([0, 4])
    vanilla_ransac_images_poses = {}

    prosac_data = np.empty([0, 4])
    prosac_images_poses = {}

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

            vanilla_ransac_images_poses[image] = best_model
            vanilla_ransac_data = np.r_[vanilla_ransac_data, np.array([inliers_no, ouliers_no, iterations, elapsed_time]).reshape([1,4])]

            # get sorted image matches
            # 6 is the lowes_distance_inverse, 7 is the heatmap value
            lowes_distances = matches_for_image[:, 6]
            heatmap_vals = matches_for_image[:, 7] / matches_for_image[:, 7].sum()
            score_list = lowes_distances * heatmap_vals # or you can use, score_list = lowes_distances

            # sorted_indices
            sorted_indices = np.argsort(score_list)
            # in descending order
            sorted_matches = matches_for_image[sorted_indices[::-1]]

            # prosac
            start = time.time()
            inliers_no, ouliers_no, iterations, best_model = prosac(sorted_matches)
            end = time.time()
            elapsed_time = end - start

            prosac_images_poses[image] = best_model
            prosac_data = np.r_[prosac_data, np.array([inliers_no, ouliers_no, iterations, elapsed_time]).reshape([1, 4])]
        else:
            print(image + " has less than 4 matches..")

    # NOTE: folders .../RANSAC_results/"+features_no+"/... where created manually..
    if(weighted):
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/vanilla_ransac_images_pose_" + str(exponential_decay_value) + "_weighted.npy", vanilla_ransac_images_poses)
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/vanilla_ransac_data_" + str(exponential_decay_value) + "_weighted.npy", vanilla_ransac_data)

        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/prosac_images_pose_" + str(exponential_decay_value) + "_weighted.npy", prosac_images_poses)
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/prosac_data_" + str(exponential_decay_value) + "_weighted.npy", prosac_data)

        print("\n")
        print("Weighted Matches Results for exponential_decay_value " + str(exponential_decay_value/10) + ":")
        print("Vanillia RANSAC")
        print("     Average Inliers: " + str(np.mean(vanilla_ransac_data[:,0])))
        print("     Average Outliers: " + str(np.mean(vanilla_ransac_data[:,1])))
        print("     Average Iterations: " + str(np.mean(vanilla_ransac_data[:,2])))
        print("     Average Time (s): " + str(np.mean(vanilla_ransac_data[:,3])))
        print("PROSAC")
        print("     Average Inliers: " + str(np.mean(prosac_data[:, 0])))
        print("     Average Outliers: " + str(np.mean(prosac_data[:, 1])))
        print("     Average Iterations: " + str(np.mean(prosac_data[:, 2])))
        print("     Average Time (s): " + str(np.mean(prosac_data[:, 3])))
        print("<---->")
    else:
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/vanilla_ransac_images_pose_" + str(exponential_decay_value) + ".npy", vanilla_ransac_images_poses)
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/vanilla_ransac_data_" + str(exponential_decay_value) + ".npy", vanilla_ransac_data)

        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/prosac_images_pose_" + str(exponential_decay_value) + ".npy", prosac_images_poses)
        np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/prosac_data_" + str(exponential_decay_value) + ".npy", prosac_data)

        print("\n")
        print("Non-Weighted Matches Results for exponential_decay_value " + str(exponential_decay_value/10) + ":")
        print("Vanillia RANSAC")
        print("     Average Inliers: " + str(np.mean(vanilla_ransac_data[:,0])))
        print("     Average Outliers: " + str(np.mean(vanilla_ransac_data[:,1])))
        print("     Average Iterations: " + str(np.mean(vanilla_ransac_data[:,2])))
        print("     Average Time (s): " + str(np.mean(vanilla_ransac_data[:,3])))
        print("PROSAC")
        print("     Average Inliers: " + str(np.mean(prosac_data[:, 0])))
        print("     Average Outliers: " + str(np.mean(prosac_data[:, 1])))
        print("     Average Iterations: " + str(np.mean(prosac_data[:, 2])))
        print("     Average Time (s): " + str(np.mean(prosac_data[:, 3])))
        print("<---->")

    print("Done!")

# NOTE: for saving files, vanilla_ransac_images_pose and vanilla_ransac_data are repeating but it does not really matter
# because run_ransac_comparison and run_prosac_comparison will save the same data regarding those.

# colmap_features_no can be "2k", "1k", "0.5k", "0.25k"
# exponential_decay can be any of 0.1 to 0.9
# print("Running RANSAC comparison against un-weighted matches")
# run_ransac_comparison("1k", 0.5)
# print("Running RANSAC comparison against weighted matches")
# run_ransac_comparison("1k", 0.5, True)

print("Running PROSAC comparison against un-weighted matches")
run_prosac_comparison("1k", 0.5)
# print("Running PROSAC comparison against weighted matches")
# run_prosac_comparison("1k", 0.5, True)


