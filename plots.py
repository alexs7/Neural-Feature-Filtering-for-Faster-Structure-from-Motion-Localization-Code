import matplotlib.pyplot as plt
import numpy as np

# Step 1: Plot FLANN matches against image name, of both model base and complete
images_localised_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/images_localised.txt"

images_localised_labels = []
with open(images_localised_path) as f:
    images_localised_labels = f.readlines()
images_localised_labels = [x.strip() for x in images_localised_labels]

# # load matching results (number of matches for each image)
# results_all = np.loadtxt('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/results_all.txt')
# results_base = np.loadtxt('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/results_base.txt')
#
# x = np.arange(len(images_localised_labels))  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# # legends
# rects1 = ax.bar(x - width/2, results_all, width, label='Against Complete Model')
# rects2 = ax.bar(x + width/2, results_base, width, label='Against Base Model')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('FLANN Matches')
# ax.set_title('No. of feature matches of base and complete model (SFM images)')
# ax.legend()
#
# fig.tight_layout()
# plt.show()

# Step 2: Plot RANSAC performance.

# structure is: inliers_no, ouliers_no, iterations, time
vanilla_ransac_data = np.loadtxt('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/vanilla_ransac_data.npy')
modified_ransac_data = np.loadtxt('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/modified_ransac_data.npy')

# 1 - inliers_no
inliers_vanilla = vanilla_ransac_data[:,0]
inliers_modified = modified_ransac_data[:,0]
# 2 - ouliers_no
ouliers_vanilla = vanilla_ransac_data[:,1]
ouliers_modified = modified_ransac_data[:,1]
# 3 - iterations
iterations_vanilla = vanilla_ransac_data[:,2]
iterations_modified = modified_ransac_data[:,2]
# 4 - time
times_vanilla = vanilla_ransac_data[:,3]
times_modified = modified_ransac_data[:,3]

breakpoint()

fig, ax = plt.subplots()
# legends
rects1 = ax.bar(x - width/2, inliers_vanilla, width, label='Vanillia RANSAC inliers')
rects2 = ax.bar(x + width/2, inliers_modified, width, label='Modified RANSAC inliers')

ax.set_ylabel('RANSAC Inliers No')
ax.set_title('No. of inliers for each RANSAC method')
ax.legend()

fig.tight_layout()
plt.show()