import numpy as np

# add slice 2 ?
result_3 = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/slice3/results.npy", allow_pickle=True).item()
result_4 = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/slice4/results.npy", allow_pickle=True).item()
result_6 = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/slice6/results.npy", allow_pickle=True).item()
result_10 = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/slice10/results.npy", allow_pickle=True).item()
result_11 = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/slice11/results.npy", allow_pickle=True).item()

# CMU
results_cmu = [result_3, result_4, result_6, result_10, result_11]

# Coop
result_coop = np.load("/home/alex/fullpipeline/colmap_data/Coop_data/slice1/results.npy", allow_pickle=True).item()

# print individual (This might not be needed for the paper)
# for results in results_cmu:
#     print("-------")
#     for k,v in results.items():
#         print(k + ": Inliers | Outliers | Iters | Time | TransError | RotError (Means)")
#         print("%2.2f & %2.2f & %2.2f & %2.2f & %2.2f & %2.2f" % (v[0], v[1], v[2], v[3], v[4], v[5] ))
#         print()

# print the average of all CMU slices

print("-------------- RANSACs/PROSAC Stats")

# all
# ransac_types = ['ransac_base', 'prosac_base', 'ransac_live', 'ransac_dist_heatmap_val', 'ransac_dist_reliability_score',
#                 'ransac_dist_visibility_score', 'inverse_lowes_ratio', 'reliability_higher_neighbour_heatmap_value', 'reliability_higher_neighbour_score',
#                 'higher_neighbour_visibility_score', 'lowes_by_reliability_score_ratio', 'lowes_by_heatmap_value_ratio',
#                 'lowes_by_higher_neighbour_reliability_score', 'lowes_by_higher_neighbour_heatmap_value']
# For paper
ransac_types = ['ransac_base', 'prosac_base', 'ransac_live', 'ransac_dist_heatmap_val',
                'ransac_dist_visibility_score', 'inverse_lowes_ratio', 'reliability_higher_neighbour_heatmap_value', 'reliability_higher_neighbour_score',
                'higher_neighbour_visibility_score', 'lowes_by_reliability_score_ratio']

print("RANSAC Types number: " + str(len(ransac_types)))

print("ransac_type: Inliers | Outliers | Iters | Time | TransError | RotError (Means of all cmu slices - Means)")
for ransac_type in ransac_types:
    inliers = []
    outliers = []
    iters = []
    time = []
    transerror = []
    roterror = []
    for results in results_cmu:
        inliers.append(results[ransac_type][0])
        outliers.append(results[ransac_type][1])
        iters.append(results[ransac_type][2])
        time.append(results[ransac_type][3])
        transerror.append(results[ransac_type][4])
        roterror.append(results[ransac_type][5])

    inliers_mean = np.array(inliers).mean()
    outliers_mean = np.array(outliers).mean()
    iters_mean = np.array(iters).mean()
    time_mean = np.array(time).mean()
    transerror_mean = np.array(transerror).mean()
    roterror_mean = np.array(roterror).mean()
    print("%2.0f & %2.0f & %2.0f & %2.2f & %2.2f & %2.2f" % (inliers_mean, outliers_mean, iters_mean, time_mean, transerror_mean, roterror_mean) + " - " + ransac_type)

print()
print("ransac_type: Inliers | Outliers | Iters | Time | TransError | RotError (Coop - Means)")
for ransac_type in ransac_types:
    inliers = []
    outliers = []
    iters = []
    time = []
    transerror = []
    roterror = []

    # the results here are already mean
    inliers_mean = result_coop[ransac_type][0]
    outliers_mean = result_coop[ransac_type][1]
    iters_mean = result_coop[ransac_type][2]
    time_mean = result_coop[ransac_type][3]
    transerror_mean = result_coop[ransac_type][4]
    roterror_mean = result_coop[ransac_type][5]

    print("%2.0f & %2.0f & %2.0f & %2.2f & %2.2f & %2.2f" % (inliers_mean, outliers_mean, iters_mean, time_mean, transerror_mean, roterror_mean) + " - " + ransac_type)

print()
print("-------------- Matches Stats")

# matches numbers
slices = ["slice3", "slice4", "slice6", "slice10", "slice11"]
print(slices)
print("Base Matches | Live Matches - (Means)")
print("CMU")
all_base_matches = []
all_live_matches = []
for slice in slices:
    total_base_matches = []
    base_matches = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/"+slice+"/matches_base.npy", allow_pickle=True).item()
    for k,v in base_matches.items():
        total_base_matches.append(v.shape[0])
    base_matches_mean = np.array(total_base_matches).sum() / len(total_base_matches)
    all_base_matches.append(base_matches_mean)

    total_live_matches = []
    live_matches = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/"+slice+"/matches_live.npy", allow_pickle=True).item()
    for k,v in live_matches.items():
        total_live_matches.append(v.shape[0])
    live_matches_mean = np.array(total_live_matches).sum() / len(total_live_matches)
    all_live_matches.append(live_matches_mean)

    print("%2.0f & %2.0f" % (base_matches_mean, live_matches_mean))

print("All CMU matches")
all_base_matches_mean = np.array(all_base_matches).mean()
all_live_matches_mean = np.array(all_live_matches).mean()
print("%2.0f & %2.0f" % (all_base_matches_mean , all_live_matches_mean))

print("Coop")
base_matches = np.load("/home/alex/fullpipeline/colmap_data/Coop_data/slice1/matches_base.npy", allow_pickle=True).item()
live_matches = np.load("/home/alex/fullpipeline/colmap_data/Coop_data/slice1/matches_live.npy", allow_pickle=True).item()
total_base_matches = []
for k, v in base_matches.items():
    total_base_matches.append(v.shape[0])
base_matches_mean = np.array(total_base_matches).sum() / len(total_base_matches)

total_live_matches = []
for k, v in live_matches.items():
    total_live_matches.append(v.shape[0])
live_matches_mean = np.array(total_live_matches).sum() / len(total_live_matches)

print("%2.0f & %2.0f" % (base_matches_mean , live_matches_mean))






