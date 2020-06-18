# applies the exponential decay on images - not 3D points as it was before!
# and also generates and saves the base VM and complete VM
# heatmap VM contains exponential decay applied (so there will be N=exponential_decay_values of those)
from load_sessions import get_sessions, get_session_weight_per_image
from point3D_loader import read_points3d_default
from query_image import read_images_binary, image_localised, load_images_from_text_file
import numpy as np

def create_vm(features_no, exponential_decay_value):

    print("-- Doing features_no " + features_no + " and exponential_decay_value: " + str(exponential_decay_value))

    # by "complete model" I mean all the frames from future sessions localised in the base model (28/03), including images from base model
    complete_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/images.bin"
    complete_model_all_images = read_images_binary(complete_model_images_path)

    complete_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/"+features_no+"/points3D.bin"
    points3D = read_points3d_default(complete_model_points3D_path)  # base model's 3D points (same length as complete as we do not add points when localising new points, but different image_ds for each point)

    # all base model images
    print("Getting the base model images..")
    base_images = read_images_binary("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/vanilla_model/2020-03-28/model/0/images.bin")
    print("     Number: " + str(len(base_images)))

    # all future session images - already localised in complete model
    print("Getting images from other future sessions.. (or models)")
    # these have to be sorted by time!
    sessions_image_sets = get_sessions()

    print("Creating VMs..")
    complete_model_visibility_matrix = np.empty([0, len(points3D)])
    vm_positive_value = 1  # if a point is seen from an image

    print("     Creating base VM..")
    base_visibility_matrix = np.empty([0, len(points3D)])

    for k, v in base_images.items():  # or images_from_28_03 here is base.
        points_row = np.zeros([len(points3D)])
        points_row = points_row.reshape([1, len(points3D)])
        point_index = 0
        for key, value in points3D.items():
            if (v.id in value.image_ids):
                points_row[0, point_index] = vm_positive_value
            point_index = point_index + 1  # move by one point (or column)
        base_visibility_matrix = np.r_[base_visibility_matrix, points_row]

    complete_model_visibility_matrix = np.r_[complete_model_visibility_matrix, base_visibility_matrix]

    print("     Creating future session VMs..")
    sessions_vm_matrices = {}  # this will have t as key and the session VM matrix as value

    t = len(sessions_image_sets)
    print("         Getting future sessions (oldest first!)")
    for session_image_set in sessions_image_sets:
        print("         Getting VM for future session: " + str(t))
        session_visibility_matrix = np.empty([0, len(points3D)])

        localised_images_no = 0
        for image_name in session_image_set:
            image_id = image_localised(image_name, complete_model_all_images)
            if (image_id != None):
                localised_images_no = localised_images_no + 1
                points_row = np.zeros([len(points3D)])
                points_row = points_row.reshape([1, len(points3D)])
                point_index = 0
                for key, value in points3D.items():
                    if (image_id in value.image_ids):
                        points_row[0, point_index] = vm_positive_value
                    point_index = point_index + 1  # move by one point (or column)
                session_visibility_matrix = np.r_[session_visibility_matrix, points_row]

        print("         For future session " + str(t) + " images localised " + str(localised_images_no) + "/" + str(len(session_image_set)))
        # print("         Session matrix rows " + str(session_visibility_matrix.shape[0]))
        t = t - 1
        sessions_vm_matrices[t] = session_visibility_matrix
        complete_model_visibility_matrix = np.r_[complete_model_visibility_matrix, session_visibility_matrix]

    # At this point you have a complete VM matrix without exponential decay
    print("Complete_model_visibility_matrix matrix rows: " + str(complete_model_visibility_matrix.shape[0]))

    np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/"+features_no+"/base_visibility_matrix.txt", base_visibility_matrix)
    np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/"+features_no+"/complete_model_visibility_matrix.txt", complete_model_visibility_matrix)

    # Now this is where you apply the exponential decay!
    print("Applying exponential decay of value: " + str(exponential_decay_value))
    session_images_weight = {}
    heatmap_matrix = np.empty([0, len(points3D)])
    N0 = vm_positive_value  # what the matrices already contain
    t1_2 = 1  # 1 day
    t = len(sessions_vm_matrices)  # start from reverse - (remember: zero based indexing here!)
    Nt = N0 * (exponential_decay_value) ** (t / t1_2)

    print("     Looking at base model vm, with t value " + str(t))
    print("     Exponential decay result (the higher the newer): " + str(Nt))
    session_images_weight[t] = Nt
    heatmap_matrix = np.where(base_visibility_matrix == vm_positive_value, Nt, 0)  # apply oldest t on oldest data first
    t = t - 1

    # sessions_vm_matrices should contain only localised images
    # Note: t here will reach 0, that is OK as the most recent model has not decayed at all (i.e is at 100)
    for k, vm in sessions_vm_matrices.items():
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
    np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/"+features_no+"/heatmap_matrix_avg_points_values_" +
               str(exponential_decay_value) + ".txt", heatmap_matrix_avg_points_values)

    # including base images (base images has to be created manually TODO: change this)
    base_images = load_images_from_text_file("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/base_images.txt")
    # NOTE: remember the weights there are normalised
    session_weight_per_image = get_session_weight_per_image(base_images, sessions_image_sets, session_images_weight)

    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/"+features_no+"/session_weight_per_image_" +
                str(exponential_decay_value) + ".npy", session_weight_per_image)

    # Note that heatmap here has the exponential decay applied the others are just binary matrices, it also contains the images from the base model and the future sessions
    np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/"+features_no+"/heatmap_matrix_" +
               str(exponential_decay_value) + ".txt", heatmap_matrix)

# NOTE: The folders are created manually under, /Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices
# colmap_features_no can be "2k", "1k", "0.5k", "0.25k"
# exponential_decay can be any of 0.1 to 0.9
create_vm("1k", 0.5)
