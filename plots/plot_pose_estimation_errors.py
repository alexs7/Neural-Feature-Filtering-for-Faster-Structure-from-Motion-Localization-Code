import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# TODO: Add pose refinement stage here ?
def plot_pose_errors(error, features_no):

    mean_error_data_vanillia = []
    mean_error_data_modified = []

    exp_decay_rates_values = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    exp_decay_rates_index = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

    data_index = 0
    upper_limit = 0
    if error == "translation":
        error_index = "t"
        upper_limit = 1.5
    if error == "rotation":
        error_index = "a"
        upper_limit = 5

    images_localised_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/images_localised_and_not_localised/" + features_no +"/images_localised.txt"

    images_localised_labels = []
    with open(images_localised_path) as f:
        images_localised_labels = f.readlines()
    images_localised_labels = [x.strip() for x in images_localised_labels]

    # Plot RANSAC pose results for each exponential decay value
    for exp_decay_rate_index in exp_decay_rates_index:
        # get the data for each features_no for both RANSAC version
        vanilla_pose_data = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/vanilla_ransac_results_"+error_index+"_"+features_no+"_"+exp_decay_rate_index+".npy")
        mean_error_data_vanillia.append(np.mean(vanilla_pose_data))

        modified_pose_data = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/modified_ransac_results_"+error_index+"_"+features_no+"_"+exp_decay_rate_index+".npy")
        mean_error_data_modified.append(np.mean(modified_pose_data))

    labels = exp_decay_rates_values
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    # first plot - for inliers
    fig, ax = plt.subplots(figsize=(18,10))

    # legends
    rects1 = ax.bar(x - width/2, mean_error_data_modified, width, label='Mean error for each image Modified RANSAC ' + error)
    rects2 = ax.bar(x + width/2, mean_error_data_vanillia, width, label='Mean error for each Vanillia RANSAC  ' + error)

    autolabel(rects1,ax)
    autolabel(rects2,ax)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number Mean for ' + error)
    ax.set_title('Pose Errors for ' + error + " and features_no " + features_no)
    ax.set_ylim(0, upper_limit)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    # plt.show()
    plt.savefig("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/plots/pose_errors/"+error+"_error_for_features_no_"+features_no+".png")

errors_to_show = ["translation", "rotation"]
colmap_features_no = ["2k", "1k", "0.5k", "0.25k"]
for error in errors_to_show:
    for features_no in colmap_features_no:
        print("Getting " + error + " errors for features_no: " + features_no)
        plot_pose_errors(error, features_no)