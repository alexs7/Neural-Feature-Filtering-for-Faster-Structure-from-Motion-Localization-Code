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
def plot_pose_errors(error, features_no, exponential_decay_value):

    mean_error_data_vanillia = 0
    mean_error_data_modified = 0

    data_index = 0
    upper_limit = 0
    if error == "translation":
        error_index = "t"
        upper_limit = 2
    if error == "rotation":
        error_index = "a"
        upper_limit = 5

    images_localised_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/images_localised_and_not_localised/" + features_no +"/images_localised.txt"

    # get the data for each features_no for both RANSAC versions
    vanilla_pose_data = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/vanilla_ransac_results_"+error_index+"_"+features_no+"_"+str(exponential_decay_value)+".npy")
    mean_error_data_vanillia = np.mean(vanilla_pose_data)
    error_bar_std_vanillia = np.std(vanilla_pose_data)

    modified_pose_data = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/modified_ransac_results_"+error_index+"_"+features_no+"_"+str(exponential_decay_value)+".npy")
    mean_error_data_modified = np.mean(modified_pose_data)
    error_bar_std_modified = np.std(modified_pose_data)

    labels = ["Modified", "Vanillia"]
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    errors = [error_bar_std_modified, error_bar_std_vanillia]
    data = [mean_error_data_modified, mean_error_data_vanillia]

    # first plot - for inliers
    fig, ax = plt.subplots()

    # legends
    rects1 = ax.bar(labels, data, yerr=errors, label='Mean error for each Vanillia RANSAC  ' + error)
    autolabel(rects1, ax)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Error for ' + error)
    ax.set_title('Pose Errors for ' + error + " and feat. " + features_no + " exp. " + str(exponential_decay_value))
    ax.set_ylim(0, upper_limit)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    # plt.show()
    plt.savefig("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/plots/pose_errors/"+error+"_error_for_features_no_"+features_no+".png")

def plot_pose_errors_weighted(error, features_no, exponential_decay_value):

    mean_error_data_vanillia = 0
    mean_error_data_modified = 0

    data_index = 0
    upper_limit = 0
    if error == "translation":
        error_index = "t"
        upper_limit = 2
    if error == "rotation":
        error_index = "a"
        upper_limit = 5

    images_localised_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/images_localised_and_not_localised/" + features_no +"/images_localised.txt"

    # get the data for each features_no for both RANSAC versions
    vanilla_pose_data = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/vanilla_ransac_results_"+error_index+"_"+features_no+"_"+str(exponential_decay_value)+"_weighted.npy")
    mean_error_data_vanillia = np.mean(vanilla_pose_data)
    error_bar_std_vanillia = np.std(vanilla_pose_data)

    modified_pose_data = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/modified_ransac_results_"+error_index+"_"+features_no+"_"+str(exponential_decay_value)+"_weighted.npy")
    mean_error_data_modified = np.mean(modified_pose_data)
    error_bar_std_modified = np.std(modified_pose_data)

    labels = ["Modified", "Vanillia"]
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    errors = [error_bar_std_modified, error_bar_std_vanillia]
    data = [mean_error_data_modified, mean_error_data_vanillia]

    # first plot - for inliers
    fig, ax = plt.subplots()

    # legends
    rects1 = ax.bar(labels, data, yerr=errors, label='Mean error for each Vanillia RANSAC  ' + error)
    autolabel(rects1, ax)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Error for ' + error)
    ax.set_title('Pose Errors for ' + error + " and feat. " + features_no + " exp. " + str(exponential_decay_value))
    ax.set_ylim(0, upper_limit)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    # plt.show()
    plt.savefig("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/plots/pose_errors/"+error+"_error_for_features_no_"+features_no+"_weighted.png")

errors_to_show = ["translation", "rotation"]
colmap_features_no = ["2k", "1k", "0.5k", "0.25k"]
for error in errors_to_show:
    print("Getting " + error + " errors for features_no: " + "1k - Un-Weighted")
    plot_pose_errors(error, "1k", 0.5)
    print("Getting " + error + " errors for features_no: " + "1k - Weighted")
    plot_pose_errors_weighted(error, "1k", 0.5)