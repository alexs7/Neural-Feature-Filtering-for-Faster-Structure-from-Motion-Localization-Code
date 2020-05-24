import matplotlib.pyplot as plt
import numpy as np

# Plot RANSAC performance graphs

mean_inliers_modified = []
mean_outliers_modified = []
mean_iterations_modified = []
mean_time_modified = []

mean_inliers_vanillia = []
mean_outliers_vanillia = []
mean_iterations_vanillia = []
mean_time_vanillia = []

indices_for_loading = np.arange(1,10)
for index in indices_for_loading:

    vanilla_ransac_data = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/vanilla_ransac_data_"+str(index)+".npy")
    mean_inliers_vanillia.append(np.mean(vanilla_ransac_data[:,0]))
    mean_outliers_vanillia.append(np.mean(vanilla_ransac_data[:,1]))
    mean_iterations_vanillia.append(np.mean(vanilla_ransac_data[:,2]))
    mean_time_vanillia.append(np.mean(vanilla_ransac_data[:,3]))

    modified_ransac_data = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/modified_ransac_data_"+str(index)+".npy")
    mean_inliers_modified.append(np.mean(modified_ransac_data[:,0]))
    mean_outliers_modified.append(np.mean(modified_ransac_data[:,1]))
    mean_iterations_modified.append(np.mean(modified_ransac_data[:,2]))
    mean_time_modified.append(np.mean(modified_ransac_data[:,3]))

labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

# first plot - for inliers
fig0, ax0 = plt.subplots()

# legends
rects1 = ax0.bar(x - width/2, mean_inliers_vanillia, width, label='Inliers Mean for Vanilla RANSAC')
rects2 = ax0.bar(x + width/2, mean_inliers_modified, width, label='Inliers Mean for Modified RANSAC')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax0.set_ylabel('Inliers Number Mean')
ax0.set_title('Modified vs Vanilla RANSAC Inliers')
ax0.set_xticks(x)
ax0.set_xticklabels(labels)
ax0.legend()

fig0.tight_layout()

# second plot - for outliers
fig1, ax1 = plt.subplots()

# legends
rects11 = ax1.bar(x - width/2, mean_outliers_vanillia, width, label='Outliers Mean for Vanilla RANSAC')
rects21 = ax1.bar(x + width/2, mean_outliers_modified, width, label='Outliers Mean for Modified RANSAC')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('Outliers Number Mean')
ax1.set_title('Modified vs Vanilla RANSAC Outliers')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()

fig1.tight_layout()

# third plot - for iterations
fig1, ax2 = plt.subplots()

# legends
rects11 = ax2.bar(x - width/2, mean_iterations_vanillia, width, label='Iterations Mean Vanilla RANSAC')
rects21 = ax2.bar(x + width/2, mean_iterations_modified, width, label='Iterations Mean Modified RANSAC')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax2.set_ylabel('Iterations Number Mean')
ax2.set_title('Modified vs Vanilla RANSAC Iterations')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()

fig1.tight_layout()

# fourth plot - for time
fig1, ax3 = plt.subplots()

# legends
rects11 = ax3.bar(x - width/2, mean_time_vanillia, width, label='Time in (s) Mean Vanilla RANSAC')
rects21 = ax3.bar(x + width/2, mean_time_modified, width, label='Time in (s) Mean Modified RANSAC')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax3.set_ylabel('Time Mean in Seconds')
ax3.set_title('Modified vs Vanilla RANSAC Time in (s)')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.legend()

fig1.tight_layout()

plt.show()



