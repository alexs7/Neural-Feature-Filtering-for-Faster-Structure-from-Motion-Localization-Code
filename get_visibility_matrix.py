# applies the exponential decay on images - not 3D points as it was before!
# and also generates and saves the base VM and complete VM
# heatmap VM contains exponential decay applied (so there will be N=exponential_decay_values of those)
from database import COLMAPDatabase
from parameters import Parameters
from point3D_loader import read_points3d_default, index_dict, index_dict_reverse
from query_image import read_images_binary, image_localised
import numpy as np

def get_row(image_id, points3D, vm_positive_value):
    point_index = 0
    points_row = np.zeros([len(points3D)]).reshape([1, len(points3D)])
    for k, v in points3D.items():
        if (image_id in v.image_ids):
            points_row[0, point_index] = vm_positive_value
        point_index += 1  # move by one point (or column)
    return points_row

# 0 MUST be base session then 1 next session etc etc...
def get_db_sessions(no_images_per_session):
    sessions = {}
    images_traversed = 0
    for i in range(len(no_images_per_session)):
        no_images = no_images_per_session[i]
        image_ids = []
        start = images_traversed
        end = start + no_images
        for k in range(start,end):
            id = k + 1 # to match db ids
            image_ids.append(id)
            images_traversed += 1
        sessions[i] = image_ids
    return sessions

def create_vm(features_no, exponential_decay_value):
    print("Creating VM for features_no " + features_no + " and exponential_decay_value: " + str(exponential_decay_value))
    # by "live model" I mean all the frames from future sessions localised in the base model, including images from base model
    live_model_all_images = read_images_binary(Parameters.live_model_images_path)
    live_model_points3D = read_points3d_default(Parameters.live_model_points3D_path)  # live model's 3D points (same length as base as we do not add points when localising new points, but different image_ds for each point)
    db = COLMAPDatabase.connect(Parameters.live_db_path)

    sessions_numbers = np.loadtxt(Parameters.no_images_per_session_path).astype(int)
    sessions_from_db = get_db_sessions(sessions_numbers)  # session_index -> [images_ids]

    print("First Loop..")
    # first loop is to get the total number of localised images in a session
    localised_images_no_per_session = []
    for session_id, session_images_ids, in sessions_from_db.items():
        image_session_idx = 0 #images local index in each session
        for image_id, image_data in sorted(live_model_all_images.items()): #db id of localised image
            if(image_id in session_images_ids):
                image_session_idx +=1
        localised_images_no_per_session.append(image_session_idx)

    print("Second Loop..")
    # second loop is to complete the metadata for sessions - can't have one loop sadly
    images_metadata = np.zeros([len(live_model_all_images), 2])
    matrix_idx = 0  # global matrix index
    for session_id, session_images_ids, in sessions_from_db.items():
        image_session_idx = 0  # images local index in each session
        for image_id, image_data in sorted(live_model_all_images.items()): #db id of localised image
            if (image_id in session_images_ids):
                image_session_idx += 1
                images_metadata[matrix_idx, 0] = image_session_idx
                images_metadata[matrix_idx, 1] = localised_images_no_per_session[session_id]
                matrix_idx += 1

    print("Third Loop..")
    # third loop is to build the binary VM matrix
    binary_visibility_matrix = np.empty([0, len(live_model_points3D)])
    for image_id, _ in sorted(live_model_all_images.items()):
        points_row = get_row(image_id, live_model_points3D, 1)
        binary_visibility_matrix = np.r_[binary_visibility_matrix, points_row]

    points3D_idx = index_dict(live_model_points3D)
    binary_visibility_matrix_cols_sum = binary_visibility_matrix.sum(axis = 0)
    reliability_scores = []

    for idx, _ in points3D_idx.items():
        current_reliability_score = binary_visibility_matrix_cols_sum[idx]
        final_reliability_score = 0
        for row_no in range(binary_visibility_matrix.shape[0]-1, -1, -1):
            elem = binary_visibility_matrix[row_no, idx]
            time = images_metadata[row_no,0]
            half_life = images_metadata[row_no,1]
            if(elem == 1):
                final_reliability_score = final_reliability_score + current_reliability_score * 0.5 ** (-time/half_life)
                current_reliability_score = current_reliability_score * 0.5 ** (-time/half_life)
            else:
                final_reliability_score = final_reliability_score + current_reliability_score * 0.5 ** (time/half_life)
                current_reliability_score = current_reliability_score * 0.5 ** (time/half_life)

        reliability_scores.append(final_reliability_score)

    reliability_scores = np.array(reliability_scores)

    N0 = 1  #default value, if a point is seen from an image
    t1_2 = 1  # 1 day
    live_model_visibility_matrix = np.empty([0, len(live_model_points3D)]) #or heatmap..
    for sessions_no, image_ids in sessions_from_db.items():
        t = len(sessions_from_db) - (sessions_no + 1) + 1 #since zero-based (14/07/2020, need to add one so it starts from the last number and goes down..)
        Nt = N0 * (exponential_decay_value) ** (t / t1_2)
        # print("t: " + str(t))
        # print("Nt: " + str(Nt))
        for image_id in image_ids:
            image_name = db.execute("SELECT name FROM images WHERE image_id = " + "'" + str(image_id) + "'")
            image_name = str(image_name.fetchone()[0])
            if(image_localised(image_name, live_model_all_images) != None):
                points_row = get_row(image_id, live_model_points3D, Nt)
                live_model_visibility_matrix = np.r_[live_model_visibility_matrix, points_row]

    # This vector will contain the points' visibility values that will be used in RANSAC dist version
    heatmap_matrix_summed_points_values = np.sum(live_model_visibility_matrix, axis=0)

    print("Saving files...")
    # reshaping them first
    reliability_scores = reliability_scores.reshape([1, reliability_scores.shape[0]])
    heatmap_matrix_summed_points_values = heatmap_matrix_summed_points_values.reshape([1, heatmap_matrix_summed_points_values.shape[0]])

    np.save(Parameters.points3D_scores_2_path, reliability_scores)
    np.save(Parameters.points3D_scores_1_path, heatmap_matrix_summed_points_values)
    # Save binary_visibility_matrix
    np.save(Parameters.binary_visibility_matrix_path, binary_visibility_matrix)

# NOTE: The folders are created manually under, /Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices
# colmap_features_no can be "2k", "1k", "0.5k", "0.25k"
# exponential_decay can be any of 0.1 to 0.9
create_vm("1k", 0.5)
