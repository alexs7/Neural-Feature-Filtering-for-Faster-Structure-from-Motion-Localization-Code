from point3D_loader import read_points3d_default
from query_image import read_images_binary, get_image_camera_center
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

# gets visibility matrix for a model
images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/images.bin"
images = read_images_binary(images_path)
points3D = read_points3d_default("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/points3D.bin")
path_to_query_images_file = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_name.txt"

# get images
with open(path_to_query_images_file) as f:
    all_query_images = f.readlines()
all_query_images = [x.strip() for x in all_query_images]

# getting the ones that have been localised
query_images = []
for query_image in all_query_images:
    camera_center = get_image_camera_center(images_path, query_image)
    if (camera_center.size != 0):
        query_images.append(query_image)

sfm_images = []
for k, v in images.items():
    if(v.name not in query_images):
        sfm_images.append(v.name)

print("SFM frame: " + str(len(sfm_images)))
print("Localised frame: " + str(len(query_images)))

# the next two loops are for sorting and creating a list of images by time!
images_dict = {}
images_dict_sorted = {}

for k, v in images.items():
    images_dict[int(v.name.split('_')[1].split(".")[0])] = v.id

for k in sorted(images_dict.keys()):
    images_dict_sorted[k] = images_dict[k]

visibility_matrix = np.empty([0, len(points3D)])

point3D_index = 0
points3D_indexing = {}
for key, value in points3D.items():
    points3D_indexing[value.id] = point3D_index
    point3D_index = point3D_index + 1

# loop throuh the time sorted frames and ids dict and create
# the visibility matrix
for k,v in images_dict_sorted.items():
    points_row = np.zeros([len(points3D)])
    points_row = points_row.reshape([1, len(points3D)])
    point_index = 0
    for key, value in points3D.items():
        if(v in value.image_ids):
            points_row[0, point_index] = 100
        point_index = point_index + 1
    visibility_matrix = np.r_[visibility_matrix, points_row]

np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/visibility_matrix_original.txt", visibility_matrix)

# helper methods:
def get_image(name, images):
    for k, v in images.items():
        if (v.name == name):
            return v

# loop through query images
previously_viewed_points = []
for query_image in query_images:

    print("Doing frame " + query_image)

    # get the point3D ids that this query image is looking at
    localised_frame_points3D_id = []
    for k,v in images.items():
        if(v.name == query_image):
            for i in range(len(v.point3D_ids)):
                if(v.point3D_ids[i] != -1):
                    localised_frame_points3D_id.append(v.point3D_ids[i])
    print(" Frame " + query_image + " looks at " + str(len(localised_frame_points3D_id)) + " points")
    localised_frame_points3D_id = np.unique(localised_frame_points3D_id) #clear duplicates

    # how many points of localised_frame_points3D do the other sfm images see ?
    images_scores = {}
    for sfm_image in sfm_images:
        image_score = 0
        for i in range(len(localised_frame_points3D_id)):
            if(localised_frame_points3D_id[i] in get_image(sfm_image, images).point3D_ids):
                image_score = image_score + 1
        images_scores[sfm_image] = image_score

    # sort by points seen, descending, so you can pick up the most prominent
    images_scores = {k: v for k, v in sorted(images_scores.items(), key=lambda item: -item[1])}

    # convert to array so you can pick up the first 5
    images_scores_list = []
    for key, value in images_scores.items():
        temp = [key, value]
        images_scores_list.append(temp)
    neighbours_no = 5
    images_scores_list = images_scores_list[0 : neighbours_no]

    # get the all 3D points that the neighbours are looking at (except the ones the localised/query frame is looking at)
    # and also make sure you haven't already seen them before in previous neighbours' frames
    localised_frame_neighbours_points3D_ids = []
    for i in range(len(images_scores_list)):
        for k,v in images.items():
            if (v.name == images_scores_list[i][0]):
                counter = 0
                for k in range(len(v.point3D_ids)):
                    if(v.point3D_ids[k] != -1 and v.point3D_ids[k] not in localised_frame_points3D_id) and v.point3D_ids[k] not in previously_viewed_points:
                        counter = counter + 1
                        localised_frame_neighbours_points3D_ids.append(v.point3D_ids[k])
                        previously_viewed_points.append(v.point3D_ids[k])
                print(" Added " + str(counter) + " points for " + v.name)

    localised_frame_neighbours_points3D_ids = np.unique(localised_frame_neighbours_points3D_ids)

    # ..now reduce their value using exponential decay
    N0 = 100
    t1_2 = 1 # 1 day (this will increase)
    t = 1
    Nt = N0 * (0.5)**(t/t1_2)
    for i in range(len(localised_frame_neighbours_points3D_ids)):
        vm_point3D_index = points3D_indexing[localised_frame_neighbours_points3D_ids[i]]
        visibility_matrix[:, vm_point3D_index] = (0.5) ** (t / t1_2) * visibility_matrix[:, vm_point3D_index]

# at this point you play with the data
# sum_over_columns = visibility_matrix.sum(axis=0)
# np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/sum_over_columns_new.txt", sum_over_columns)
np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/visibility_matrix_new.txt", visibility_matrix)
