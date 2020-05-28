import matplotlib.pyplot as plt
import numpy as np

# Plot RANSAC pose results for each exponential decay value
# TODO: Add pose refinement stage here ?
images_localised_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/images_localised.txt"

images_localised_labels = []
with open(images_localised_path) as f:
    images_localised_labels = f.readlines()
images_localised_labels = [x.strip() for x in images_localised_labels]

mean_values_t_modified = []
mean_values_t_vanilla = []
mean_values_a_modified = []
mean_values_a_vanilla = []

indices_for_loading = np.arange(1,10)
for index in indices_for_loading:

    # modified and vanilla ransac
    # translation errors
    vanilla_ransac_results_t = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/vanilla_ransac_results_t_"+str(index)+".npy")
    mean_values_t_vanilla.append(np.mean(vanilla_ransac_results_t))

    modified_ransac_results_t = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/modified_ransac_results_t_"+str(index)+".npy")
    mean_values_t_modified.append(np.mean(modified_ransac_results_t))

    # rotation errors
    vanilla_ransac_results_a = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/vanilla_ransac_results_a_"+str(index)+".npy")
    mean_values_a_vanilla.append(np.mean(vanilla_ransac_results_a))

    modified_ransac_results_a = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/modified_ransac_results_a_"+str(index)+".npy")
    mean_values_a_modified.append(np.mean(modified_ransac_results_a))


labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

# first plot - for translations
fig0, ax0 = plt.subplots()

# legends
rects1 = ax0.bar(x - width/2, mean_values_t_vanilla, width, label='Translation Errors Vanilla RANSAC')
rects2 = ax0.bar(x + width/2, mean_values_t_modified, width, label='Translation Errors Modified RANSAC')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax0.set_ylabel('Translation Errors in COLMAP Units')
ax0.set_title('Modified vs Vanilla RANSAC')
ax0.set_xticks(x)
ax0.set_xticklabels(labels)
ax0.legend()

fig0.tight_layout()

# second plot - for rotations
fig1, ax1 = plt.subplots()

# legends
rects11 = ax1.bar(x - width/2, mean_values_a_vanilla, width, label='Rotation Errors Vanilla RANSAC')
rects21 = ax1.bar(x + width/2, mean_values_a_modified, width, label='Rotation Errors Modified RANSAC')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('Rotation Errors in radians')
ax1.set_title('Modified vs Vanilla RANSAC')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()

fig1.tight_layout()

plt.show()