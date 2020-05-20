import matplotlib.pyplot as plt
import numpy as np

# Step 1: Plot FLANN matches against image name, of both model base and complete
# images_localised_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/images_localised.txt"
#
# images_localised_labels = []
# with open(images_localised_path) as f:
#     images_localised_labels = f.readlines()
# images_localised_labels = [x.strip() for x in images_localised_labels]
#
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

# Step 2: Plot RANSAC performance. - no graphs here

# structure is: inliers_no, ouliers_no, iterations, time
# vanilla_ransac_data = np.loadtxt('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/vanilla_ransac_data.npy')
# modified_ransac_data = np.loadtxt('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/modified_ransac_data.npy')
#
# # 1 - inliers_no
# inliers_vanilla = vanilla_ransac_data[:,0]
# inliers_modified = modified_ransac_data[:,0]
# print("Inliers vanilla no: " + str(int(np.sum(inliers_vanilla))))
# print("Inliers modified: " + str(int(np.sum(inliers_modified))))
#
# # 2 - ouliers_no
# ouliers_vanilla = vanilla_ransac_data[:,1]
# ouliers_modified = modified_ransac_data[:,1]
# print("Outliers vanilla no: " + str(int(np.sum(ouliers_vanilla))))
# print("Outliers modified: " + str(int(np.sum(ouliers_modified))))
#
# # 3 - iterations
# iterations_vanilla = vanilla_ransac_data[:,2]
# iterations_modified = modified_ransac_data[:,2]
# print("Iterations vanilla no: " + str(int(np.sum(iterations_vanilla))))
# print("Iterations modified: " + str(int(np.sum(iterations_modified))))
#
# # 4 - time
# times_vanilla = vanilla_ransac_data[:,3]
# times_modified = modified_ransac_data[:,3]
# print("Duration (s) vanilla no: " + str(int(np.sum(times_vanilla))))
# print("Duration (s) modified: " + str(int(np.sum(times_modified))))

#  Step 3: plot as in step 1 but groups in days / sessions
# images_localised_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/images_localised.txt"
# all_sessions_dic = np.load('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/sessions_data/images_names/all_sessions_dic.npy')
# all_sessions_dic = all_sessions_dic.item()
#
# images_localised_labels = []
# with open(images_localised_path) as f:
#     images_localised_labels = f.readlines()
# images_localised_labels = [x.strip() for x in images_localised_labels]
#
# # load matching results (number of matches for each image)
# results_all = np.load('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/results_all.npy').item()
# results_base = np.load('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/results_base.npy').item()
#
# results_all_over_sessions = []
# results_base_over_sessions = []
#
# for session_name, session_images in all_sessions_dic.items():
#     temp_session_result_all = []
#     temp_session_result_base = []
#     for image_name, matches_no in results_all.items():
#         if( image_name in session_images): # i.e is in current session
#             temp_session_result_all.append(matches_no)
#     for image_name, matches_no in results_base.items():
#         if( image_name in session_images): # i.e is in current session
#             temp_session_result_base.append(matches_no)
#
#
#     results_all_over_sessions.append(np.sum(temp_session_result_all))
#     results_base_over_sessions.append(np.sum(temp_session_result_base))
#
# x = np.arange(len(all_sessions_dic.keys()))  # the label locations
# labels = all_sessions_dic.keys()
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# # legends
# rects1 = ax.bar(x - width/2, results_all_over_sessions, width, label='Against Complete Model')
# rects2 = ax.bar(x + width/2, results_base_over_sessions, width, label='Against Base Model')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('FLANN Matches - for each day')
# ax.set_title('No. of feature matches of base and complete model (SFM images)')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
#
# # fig.tight_layout()
# plt.xticks(rotation=90)
# fig.tight_layout()
# plt.show()

# Step 4: Plot the pose errors
images_localised_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/images_localised.txt"

images_localised_labels = []
with open(images_localised_path) as f:
    images_localised_labels = f.readlines()
images_localised_labels = [x.strip() for x in images_localised_labels]

# modified and vanilla ransac
# translation errors
vanilla_ransac_results_t = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/vanilla_ransac_results_t.npy")
modified_ransac_results_t = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/modified_ransac_results_t.npy")
# rotation errors
vanilla_ransac_results_a = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/vanilla_ransac_results_a.npy")
modified_ransac_results_a = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/modified_ransac_results_a.npy")

x = np.arange(len(images_localised_labels))  # the label locations
width = 0.35  # the width of the bars

fig0, ax0 = plt.subplots()
# legends
rects1 = ax0.bar(x - width/2, vanilla_ransac_results_t, width, label='Translation Errors Vanilla RANSAC')
rects2 = ax0.bar(x + width/2, modified_ransac_results_t, width, label='Translation Errors Modified RANSAC')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax0.set_ylabel('Translation Errors in COLMAP Units')
ax0.set_title('Modified vs Vanilla RANSAC')
ax0.legend()

fig0.tight_layout()

fig1, ax1 = plt.subplots()
# legends
rects11 = ax1.bar(x - width/2, vanilla_ransac_results_a, width, label='Rotation Errors Vanilla RANSAC')
rects21 = ax1.bar(x + width/2, modified_ransac_results_a, width, label='Rotation Errors Modified RANSAC')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('Rotation Errors in radians')
ax1.set_title('Modified vs Vanilla RANSAC')
ax1.legend()

fig1.tight_layout()

plt.show()