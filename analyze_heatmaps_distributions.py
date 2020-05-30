# this file will be used to analyze and extract information from each heatmap's
# distribution. Remember each distribution is the average of each point column and then
# normalised so it adds up to 1.
# TODO: Needs revising!

# import numpy as np
#
# exponential_decay_values = np.linspace(0,1, num=10, endpoint=False)[1:10]
#
# for i in range(len(exponential_decay_values)):
#     index = i + 1 #to match the file naming.. 1 is for 0.1, 2 is for 0.2 etc etc (exponential_decay_value)
#     dist = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/heatmap_matrix_avg_points_values_"+str(index)+".txt")
#     dist = dist.reshape([1,dist.shape[0]])
#     mean = np.mean(dist)
#     dist = np.where(dist > mean, dist, 0)
#     dist = dist / np.sum(dist)
#     np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/heatmap_matrix_avg_points_values_"+str(index)+"_mean_distri.txt", dist)
#
# print("Done!")
#


