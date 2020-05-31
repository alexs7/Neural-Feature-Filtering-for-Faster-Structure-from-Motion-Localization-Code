import matplotlib.pyplot as plt
import numpy as np

# Plot RANSAC performance graphs
def plot_ransac(data, features_no):

    mean_data_vanillia = []
    mean_data_modified = []

    exp_decay_rates_values = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    exp_decay_rates_index = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

    data_index = 0
    upper_limit = 0
    if data == "inliers":
        data_index = 0
        upper_limit = 120
    if data == "outliers":
        data_index = 1
        upper_limit = 110
    if data == "avg_iters":
        data_index = 2
        upper_limit = 120
    if data == "avg_time":
        data_index = 3
        upper_limit = 0.5

    for exp_decay_rate_index in exp_decay_rates_index:
        # get the data for each features_no for both RANSAC version
        vanilla_ransac_data = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/vanilla_ransac_data_"+exp_decay_rate_index+".npy")
        mean_data_vanillia.append(np.mean(vanilla_ransac_data[:,data_index]))

        modified_ransac_data = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/"+features_no+"/modified_ransac_data_"+exp_decay_rate_index+".npy")
        mean_data_modified.append(np.mean(modified_ransac_data[:,data_index]))

    labels = exp_decay_rates_values
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    # first plot - for inliers
    fig, ax = plt.subplots(figsize=(18,10))

    # legends
    rects1 = ax.bar(x - width/2, mean_data_modified, width, label='Mean for Modified RANSAC ' + data)
    rects2 = ax.bar(x + width/2, mean_data_vanillia, width, label='Mean for Vanilla RANSAC ' + data)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number Mean for ' + data)
    ax.set_title('Modified vs Vanilla RANSAC for ' + data + " and features_no " + features_no)
    ax.set_ylim(0, upper_limit)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    # plt.show()
    plt.savefig("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/plots/"+data+"_for_features_no_"+features_no+".png")

# NOTE: remember time is in seconds
data_to_show = ["inliers", "outliers", "avg_iters", "avg_time"]
colmap_features_no = ["2k", "1k", "0.5k", "0.25k"]
for data in data_to_show:
    for features_no in colmap_features_no:
        print("Getting " + data + " for features_no: " + features_no)
        plot_ransac(data, features_no)