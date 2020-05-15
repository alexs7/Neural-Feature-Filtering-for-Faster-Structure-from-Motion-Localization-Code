import numpy as np
from ransac import run_ransac

# load localised images names
localised_images = []
path_to_query_images_file = '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/images_localised.txt'
with open(path_to_query_images_file) as f:
    localised_images = f.readlines()
localised_images = [x.strip() for x in localised_images]

matches_all = np.load('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/matches_all.npy')

for image in localised_images:
    print("Doing image: " + str(image))
    matches_for_image = matches_all.item()[image]
    print("     Number of matches: " + str(len(matches_for_image)))

    # run RANSAC here
    inliers_no, ouliers_no, iterations, best_model = run_ransac(matches_for_image)
    print("     inliers_no: " + str(inliers_no))
    print("     ouliers_no: " + str(ouliers_no))
    print("     iterations: " + str(iterations))


