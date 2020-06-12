import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.4f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# TODO: Add pose refinement stage here - no add in pose evaluator maybe ?
def plot_pose_errors(features_no, exponential_decay_value,
                     v_errors_path, m_errors_path,
                     out_path, error_string):

    # get the data for each features_no for both RANSAC versions
    v_errors = np.load(v_errors_path)
    standard_error_data_vanillia = np.mean(v_errors) / np.sqrt(v_errors.shape[0])
    error_bar_std_vanillia = np.std(v_errors)

    m_errors = np.load(m_errors_path)
    standard_error_data_modified = np.mean(m_errors) / np.sqrt(m_errors.shape[0])
    error_bar_std_modified = np.std(m_errors)

    labels = ["Modified", "Vanillia"]
    x = np.arange(len(labels))  # the label locations
    data = [standard_error_data_modified, standard_error_data_vanillia]

    # first plot - for inliers
    fig, ax = plt.subplots()

    # legends
    rects1 = ax.bar(labels, data, label='Standard Error for ' + error_string)
    autolabel(rects1, ax)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Standard Error')
    ax.set_title('Errors for ' + error_string + " and feat. " + features_no + " exp. " + str(exponential_decay_value))
    ax.set_ylim(0, 0.5) #change this accordingly
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    # plt.show()
    plt.savefig(out_path)

# colmap_features_no can be "2k", "1k", "0.5k", "0.25k"
# exponential_decay can be any of 0.1 to 0.9
features_no = "1k"
exponential_decay_value = 0.5

v_trans_errors_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/vanillia_results_t_"+features_no+"_"+str(exponential_decay_value)+".npy"
v_rotation_errors_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/vanillia_results_a_"+features_no+"_"+str(exponential_decay_value)+".npy"
m_trans_errors_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/modified_results_t_"+features_no+"_"+str(exponential_decay_value)+".npy"
m_rotation_errors_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/modified_results_a_"+features_no+"_"+str(exponential_decay_value)+".npy"

out_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/plots/pose_errors/trans_error_for_features_no_"+features_no+".png"
plot_pose_errors(features_no, exponential_decay_value, v_trans_errors_path, m_trans_errors_path, out_path, 'translations')

out_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/plots/pose_errors/rotation_error_for_features_no_"+features_no+".png"
plot_pose_errors(features_no, exponential_decay_value, v_rotation_errors_path, m_rotation_errors_path, out_path, 'rotations')