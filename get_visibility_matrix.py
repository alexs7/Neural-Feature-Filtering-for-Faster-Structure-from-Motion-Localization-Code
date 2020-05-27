# applies the exponential decay on images - not 3D points as it was before!
# and also generates and saves the base VM and complete VM
# heatmap VM contains exponential decay applied (so there will be N=exponential_decay_values of those)
from load_sessions import get_sessions
from point3D_loader import read_points3d_default
from query_image import read_images_binary, image_localised
import numpy as np

# by "complete model" I mean all the frames from future sessions localised in the base model (28/03)
complete_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/images.bin"
complete_model_all_images = read_images_binary(complete_model_images_path)
complete_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/points3D.bin"
points3D = read_points3d_default(complete_model_points3D_path) # base model's 3D points

# all base model images
print("Getting the base model images..")
images_from_28_03 = read_images_binary("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/model/0/images.bin")
print("     Number: "+ str(len(images_from_28_03)))

# all future session images - already localised in complete model
print("Getting images from other future sessions.. (or models)")
# these have to be sorted by time!
sessions_image_sets = get_sessions()

print("Creating VMs..")
complete_model_visibility_matrix = np.empty([0, len(points3D)])
vm_positive_value = 1 # if a point is seen from an image

print("     Creating base VM..")
base_visibility_matrix = np.empty([0, len(points3D)])

for k,v in images_from_28_03.items(): # or images_from_28_03 here is base.
    points_row = np.zeros([len(points3D)])
    points_row = points_row.reshape([1, len(points3D)])
    point_index = 0
    for key, value in points3D.items():
        if(v.id in value.image_ids):
            points_row[0, point_index] = vm_positive_value
        point_index = point_index + 1 # move by one point (or column)
    base_visibility_matrix = np.r_[base_visibility_matrix, points_row]

complete_model_visibility_matrix = np.r_[complete_model_visibility_matrix, base_visibility_matrix]

print("     Creating future session VMs..")
sessions_vm_matrices = {} # this will have t as key and the session VM matrix as value

t = len(sessions_image_sets)
print("         Getting future sessions (oldest first)")
for session_image_set in sessions_image_sets:
    print("         Getting VM for future session: " + str(t))
    session_visibility_matrix = np.empty([0, len(points3D)])

    localised_images_no = 0
    for image_name in session_image_set:
        image_id = image_localised(image_name, complete_model_all_images)
        if(image_id != None):
            localised_images_no = localised_images_no + 1
            points_row = np.zeros([len(points3D)])
            points_row = points_row.reshape([1, len(points3D)])
            point_index = 0
            for key, value in points3D.items():
                if (image_id in value.image_ids):
                    points_row[0, point_index] = vm_positive_value
                point_index = point_index + 1  # move by one point (or column)
            session_visibility_matrix = np.r_[session_visibility_matrix, points_row]

    print("         For future session " + str(t) + " images localised " + str(localised_images_no) +"/"+ str(len(session_image_set)) )
    # print("         Session matrix rows " + str(session_visibility_matrix.shape[0]))
    t = t - 1
    sessions_vm_matrices[t] = session_visibility_matrix
    complete_model_visibility_matrix = np.r_[complete_model_visibility_matrix, session_visibility_matrix]

# At this point you have a complete VM matrix without exponential decay
print("Complete_model_visibility_matrix matrix rows: " + str(complete_model_visibility_matrix.shape[0]))

np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/base_visibility_matrix.txt",
    base_visibility_matrix)
np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/complete_model_visibility_matrix.txt",
    complete_model_visibility_matrix)

# Now this is where you apply the exponential decay!
# try different exponential values..
# Note: that the higer the exponential_decay_value the slower it decays...
exponential_decay_values = np.linspace(0,1, num=10, endpoint=False)[1:10]

index_for_file_saving = 1 # 1 is for 0.1, 2 is for 0.2 etc etc
for exponential_decay_value in exponential_decay_values:

    print("Applying exponential decay of value: " + str(exponential_decay_value))
    session_images_weight = {}
    heatmap_matrix = np.empty([0, len(points3D)])
    N0 = vm_positive_value # what the matrices already contain
    t1_2 = 1 # 1 day
    t = len(sessions_vm_matrices) # start from reverse - (zero based indexing here!)
    Nt = N0 * (exponential_decay_value) ** (t / t1_2)

    print("     Looking at base model vm, with t value " + str(t))
    print("     Exponential decay result (the higher the newer): " + str(Nt))
    session_images_weight[t] = Nt
    heatmap_matrix = np.where(base_visibility_matrix == vm_positive_value, Nt, 0) #apply oldest t on oldest data first
    t = t-1

    # sessions_vm_matrices should contain only localised images
    # Note: t here will reach 0, that is OK as the most recent model has not decayed at all (i.e is at 100)
    for k , vm in sessions_vm_matrices.items():
        Nt = N0 * (exponential_decay_value) ** (t / t1_2)
        print("     Looking at session vm " + str(k) + " and t value is at " + str(t))
        session_images_weight[t] = Nt
        print("     Exponential decay result (the higher the newer): " + str(Nt))
        vm = np.where(vm == vm_positive_value, Nt, 0)
        heatmap_matrix = np.r_[heatmap_matrix, vm]
        t = t - 1

    # This vector will contain the points' visibility values averaged that will be used in RANSAC
    heatmap_matrix_avg_points_values = np.mean(heatmap_matrix, axis=0)
    heatmap_matrix_avg_points_values = heatmap_matrix_avg_points_values / np.sum(heatmap_matrix_avg_points_values)
    # at this point you have now a distribution (i.e sum to 1) in heatmap_matrix_avg_points_values

    print("Saving files...")
    np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/heatmap_matrix_avg_points_values_"+str(index_for_file_saving)+".txt", heatmap_matrix_avg_points_values)
    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/session_images_weight_"+str(index_for_file_saving)+".npy", session_images_weight)
    # Note that heatmap here has the exponential decay applied the others are just binary matrices, it also contains the images from the base model and the future sessions
    np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/heatmap_matrix_"+str(index_for_file_saving)+".txt", heatmap_matrix)

    index_for_file_saving = index_for_file_saving + 1

# This is still WIP
# print("Applying Set Cover Problem..")
# points3D = #TODO: point it to the base model ones!! otherwise you endup with more image ids per point!
# # create an index and point_id relationship - reverse here!
# point3D_index = 0
# points3D_indexing = {}
# for key, value in points3D.items():
#     points3D_indexing[point3D_index] = value.id
#     point3D_index = point3D_index + 1
#
# # create an index and image_id relationship
# image_index = 0
# images_indexing = {}
# for key, value in images_from_28_03.items():
#     images_indexing[value.id] = image_index
#     image_index = image_index + 1

# removed_images_ids = []
# selected_points_ids = []
# K = 100
#
# while(np.sum(base_visibility_matrix) != 0):
#     sum_points_across_images = np.sum(base_visibility_matrix, 0)
#     max_point_index = np.argmax(sum_points_across_images)
#     selected_points_ids.append(points3D_indexing[max_point_index])
#
#     images_ids_that_observe_current_max_point = np.unique(points3D[points3D_indexing[max_point_index]].image_ids)
#
#     for image_id in images_ids_that_observe_current_max_point:
#         if(np.sum(base_visibility_matrix[images_indexing[image_id],:]) >= K):
#             print("Removing image with id " + str(image_id) + " and index " + str(images_indexing[image_id]))
#             base_visibility_matrix[images_indexing[image_id], :] = 0
#
#     print("Removing point with id " + str(points3D_indexing[max_point_index]) + " and index " + str(max_point_index) + " and value " + str(sum_points_across_images[max_point_index]))
#     base_visibility_matrix[:,max_point_index] = 0
#
#     print("Sum of base_visibility_matrix: " + str(np.sum(base_visibility_matrix)))
#
# !Set Cover Problem

# #         Getting the image details from the images
#
#
# # create an index and point_id relationship
# point3D_index = 0
# points3D_indexing = {}
# for key, value in points3D.items():
#     points3D_indexing[value.id] = point3D_index
#     point3D_index = point3D_index + 1
#
# # loop throuh the time sorted frames and ids dict and create the visibility matrix
# for k,v in images_dict_sorted.items():
#     points_row = np.zeros([len(points3D)])
#     points_row = points_row.reshape([1, len(points3D)])
#     point_index = 0
#     for key, value in points3D.items():
#         if(v in value.image_ids):
#             points_row[0, point_index] = 100
#         point_index = point_index + 1 # move by one point (or column)
#     visibility_matrix = np.r_[visibility_matrix, points_row]
#
# np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/visibility_matrix_original.txt", visibility_matrix)
#
#
#
# # loop through query images
# previously_viewed_points = []
#
#
# for query_image in images_from_29_03_names:
#     if(get_image_camera_center(images_path, query_image).size != 0): #if the images was localised against the base model
#         for vm_image in images.items():
#             breakpoint()

# t = 0
# for image_set in image_sets:
#     t = t + 1
#     print("t value: " + str(t))
#
#     for query_image in image_set:
#         camera_center = get_image_camera_center(images_path, query_image)
#         if (camera_center.size == 0):
#             print(query_image + " not localised..")
#             continue
#
#         print("Doing frame " + query_image)
#         # get the point3D ids that this query image is looking at
#         localised_frame_points3D_id = []
#         for k,v in images.items():
#             if(v.name == query_image):
#                 for i in range(len(v.point3D_ids)):
#                     if(v.point3D_ids[i] != -1):
#                         localised_frame_points3D_id.append(v.point3D_ids[i])
#         print(" Frame " + query_image + " looks at " + str(len(localised_frame_points3D_id)) + " points")
#         localised_frame_points3D_id = np.unique(localised_frame_points3D_id) #clear duplicates
#
#         # how many points of the localised_frame_points3D do the the other sfm_images see ?
#         images_scores = {}
#         for sfm_image in sfm_images:
#             image_score = 0
#             for i in range(len(localised_frame_points3D_id)):
#                 if(localised_frame_points3D_id[i] in get_image(sfm_image, images).point3D_ids):
#                     image_score = image_score + 1
#             images_scores[sfm_image] = image_score
#
#         # sort by points seen, descending, so you can pick up the most prominent
#         images_scores = {k: v for k, v in sorted(images_scores.items(), key=lambda item: -item[1])}
#
#         # convert to array so you can pick up the first N
#         images_scores_list = []
#         for key, value in images_scores.items():
#             temp = [key, value]
#             images_scores_list.append(temp)
#         neighbours_no = 3 #N
#         images_scores_list = images_scores_list[0 : neighbours_no]
#
#         # get the all 3D points that the neighbours are looking at (except the ones the localised/query frame is looking at)
#         # and also make sure you haven't already seen them before in previous neighbours' frames
#         localised_frame_neighbours_points3D_ids = []
#         for i in range(len(images_scores_list)):
#             for k,v in images.items():
#                 if (v.name == images_scores_list[i][0]):
#                     counter = 0
#                     for k in range(len(v.point3D_ids)):
#                         if(v.point3D_ids[k] != -1 and v.point3D_ids[k] not in localised_frame_points3D_id and v.point3D_ids[k] not in previously_viewed_points):
#                             counter = counter + 1
#                             localised_frame_neighbours_points3D_ids.append(v.point3D_ids[k])
#                             previously_viewed_points.append(v.point3D_ids[k])
#                     print(" Added " + str(counter) + " points for " + v.name)
#
#         localised_frame_neighbours_points3D_ids = np.unique(localised_frame_neighbours_points3D_ids) #TODO: might not need this ?
#
#         # ..now reduce their value using exponential decay
#         N0 = 100
#         t1_2 = 1 # 1 day
#         Nt = N0 * (0.5)**(t/t1_2)
#
#         for i in range(len(localised_frame_neighbours_points3D_ids)):
#             vm_point3D_index = points3D_indexing[localised_frame_neighbours_points3D_ids[i]]
#             visibility_matrix[:, vm_point3D_index] = (0.5) ** (t / t1_2) * visibility_matrix[:, vm_point3D_index]
#
# # at this point you play with the data
# # sum_over_columns = visibility_matrix.sum(axis=0) #TODO: Revise this ?
# np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/visibility_matrix_new.txt", visibility_matrix)
#
# print("Done!")