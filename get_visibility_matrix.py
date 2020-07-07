# applies the exponential decay on images - not 3D points as it was before!
# and also generates and saves the base VM and complete VM
# heatmap VM contains exponential decay applied (so there will be N=exponential_decay_values of those)
from database import COLMAPDatabase
from parameters import Parameters
from point3D_loader import read_points3d_default, index_dict
from query_image import read_images_binary, image_localised, load_images_from_text_file
import numpy as np

def get_row(image_id, points3D, vm_positive_value):
    point_index = 0
    points_row = np.zeros([len(points3D)]).reshape([1, len(points3D)])
    for k, v in points3D.items():
        if (image_id in v.image_ids):
            points_row[0, point_index] = vm_positive_value
        point_index += 1  # move by one point (or column)
    return points_row

def get_db_sessions():
    # number of images per session. This is hardcoded for now, but since images are sorted by name, i.e by time in the database,
    # then you can use these numbers to get images from each session. The numbers need to be sorted by session though. First is number of base model images.
    no_images_per_session = Parameters.no_images_per_session
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
    live_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/images.bin"
    live_model_all_images = read_images_binary(live_model_images_path)

    live_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/points3D.bin"
    points3D = read_points3d_default(live_model_points3D_path)  # base model's 3D points (same length as live as we do not add points when localising new points, but different image_ds for each point)

    db_path = Parameters.db_path
    db = COLMAPDatabase.connect(db_path)

    sessions_from_db = get_db_sessions() # session_index -> [images_ids]
    binary_visibility_matrix = np.empty([0, len(points3D)])
    # This will hold information, about the binary_visibility_matrix images (rows)
    # Each row, (image) has to hold information about its position in the session,
    # and the total images in its session. Note that only localised images,
    # are in sessions hence binary_visibility_matrix.rows < than images in sessions_from_db
    # COLMAP database also holds images that are NOT localised.
    # [matrix_index] -> (session_id_position)
    binary_visibility_matrix_frames = {}
    matrix_index = 0
    for sessions_idx in range(len(sessions_from_db)):
        image_ids = sessions_from_db[sessions_idx]
        image_session_position = 1
        for id in image_ids:
            image_name = db.execute("SELECT name FROM images WHERE image_id = " + "'" + str(id) + "'")
            image_name = str(image_name.fetchone()[0])
            if (image_localised(image_name, live_model_all_images) != None):
                binary_visibility_matrix_frames[matrix_index] = (image_session_position,)
                points_row = get_row(id, points3D, 1)
                binary_visibility_matrix = np.r_[binary_visibility_matrix, points_row]
                matrix_index += 1
                image_session_position += 1
        # add the no of total images of session to each row
        total_session_images = image_session_position-1
        for idx, _ in binary_visibility_matrix_frames.items():
            binary_visibility_matrix_frames[idx] = binary_visibility_matrix_frames[idx] + (total_session_images,)

    points3D_idx = index_dict(points3D)
    binary_visibility_matrix_cols_sum = binary_visibility_matrix.sum(axis = 0)
    reliability_scores = []
    for idx, _ in points3D_idx.items():
        current_reliability_score = binary_visibility_matrix_cols_sum[idx]
        final_reliability_scores = 0
        for row_no in range(binary_visibility_matrix.shape[0]):
            elem = binary_visibility_matrix[row_no, idx]
            time = binary_visibility_matrix_frames[row_no][0]
            half_life = binary_visibility_matrix_frames[row_no][1]
            # print(" Row no %d, time/half_life: %d/%d" % (row_no, time, half_life))
            if(elem == 1):
                final_reliability_scores += current_reliability_score * 0.5 ** (-time/half_life)
            else:
                final_reliability_scores += current_reliability_score * 0.5 ** (time/half_life)
        reliability_scores.append(final_reliability_scores)

    reliability_scores = np.array(reliability_scores)

    N0 = 1  #default value, if a point is seen from an image
    t1_2 = 1  # 1 day
    live_model_visibility_matrix = np.empty([0, len(points3D)]) #or heatmap..
    for sessions_no, image_ids in sessions_from_db.items():
        t = len(sessions_from_db) - (sessions_no + 1) #since zero-based
        Nt = N0 * (exponential_decay_value) ** (t / t1_2)
        # print("t: " + str(t))
        # print("Nt: " + str(Nt))
        for id in image_ids:
            image_name = db.execute("SELECT name FROM images WHERE image_id = " + "'" + str(id) + "'")
            image_name = str(image_name.fetchone()[0])
            if(image_localised(image_name, live_model_all_images) != None):
                points_row = get_row(id, points3D, Nt)
                live_model_visibility_matrix = np.r_[live_model_visibility_matrix, points_row]

    print("Getting session weights for each image..")
    session_weight_per_image = {}
    for sessions_no, image_ids in sessions_from_db.items():
        t = len(sessions_from_db) - (sessions_no + 1) #since zero-based
        Nt = N0 * (exponential_decay_value) ** (t / t1_2)
        for id in image_ids:
            image_name = db.execute("SELECT name FROM images WHERE image_id = " + "'" + str(id) + "'")
            image_name = str(image_name.fetchone()[0])
            if(image_localised(image_name, live_model_all_images) != None):
                session_weight_per_image[image_name] = Nt

    # This vector will contain the points' visibility values that will be used in RANSAC dist version
    heatmap_matrix_summed_points_values = np.sum(live_model_visibility_matrix, axis=0)

    print("Saving files...")
    # reshaping them first
    reliability_scores = reliability_scores.reshape([1, reliability_scores.shape[0]])
    heatmap_matrix_summed_points_values = heatmap_matrix_summed_points_values.reshape([1, heatmap_matrix_summed_points_values.shape[0]])
    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/"+features_no+"/reliability_scores_" + str(exponential_decay_value) + ".npy", reliability_scores)
    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/"+features_no+"/heatmap_matrix_avg_points_values_" + str(exponential_decay_value) + ".npy", heatmap_matrix_summed_points_values)
    # NOTE: remember the weights there are normalised
    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/"+features_no+"/session_weight_per_image_" + str(exponential_decay_value) + ".npy", session_weight_per_image)

# NOTE: The folders are created manually under, /Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices
# colmap_features_no can be "2k", "1k", "0.5k", "0.25k"
# exponential_decay can be any of 0.1 to 0.9
create_vm("1k", 0.5)
