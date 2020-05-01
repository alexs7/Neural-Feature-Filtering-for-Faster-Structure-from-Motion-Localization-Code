from point3D_loader import read_points3d_default
from query_image import read_images_binary
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

# gets visibility matrix for a model
images = read_images_binary("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/images.bin")
points3D = read_points3d_default("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/points3D.bin")
path_to_query_images_file = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_name.txt"

# get images
with open(path_to_query_images_file) as f:
    query_images = f.readlines()
query_images = [x.strip() for x in query_images]

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

# loop through query images
for query_image in query_images:

    print("Doing frame " + query_image)

    localised_frame_points3D_id = []
    for k,v in images.items():
        if(v.name == query_image):
            for i in range(len(v.point3D_ids)):
                if(v.point3D_ids[i] != -1):
                    localised_frame_points3D_id.append(v.point3D_ids[i])
    print(" Frame " + query_image + " looks at " + str(len(localised_frame_points3D_id)) + " points")
    localised_frame_points3D_id = np.unique(localised_frame_points3D_id) #clear duplicates

    # how many points of localised_frame_points3D do the other images see ?
    images_scores = {}
    for k,v in images.items():
        if(v.name not in query_images): # exclude query image(s)
            image_score = 0
            for i in range(len(localised_frame_points3D_id)):
                if(localised_frame_points3D_id[i] in v.point3D_ids):
                    image_score = image_score + 1
            images_scores[v.name] = image_score

    # sort by points seen, descending
    images_scores = {k: v for k, v in sorted(images_scores.items(), key=lambda item: -item[1])}

    # convert to array so you can pick up the first 5
    images_scores_list = []
    for key, value in images_scores.items():
        temp = [key, value]
        images_scores_list.append(temp)
    neighbours_no = 5
    images_scores_list = images_scores_list[0 : neighbours_no]

    # get the all 3D points that the neighbours are looking at
    localised_frame_neighbours_points3D_ids = []
    for i in range(len(images_scores_list)):
        for k,v in images.items():
            if (v.name == images_scores_list[i][0]):
                for k in range(len(v.point3D_ids)):
                    if(v.point3D_ids[k] != -1 and v.point3D_ids[k] not in localised_frame_points3D_id):
                        localised_frame_neighbours_points3D_ids.append(v.point3D_ids[k])
                print(" Added points for " + v.name)

    localised_frame_neighbours_points3D_ids = np.unique(localised_frame_neighbours_points3D_ids)

    N0 = 100
    t1_2 = 1 # 1 day
    t = 1
    Nt = N0 * (0.5)**(t/t1_2)
    for i in range(len(localised_frame_neighbours_points3D_ids)):
        vm_point3D_index = points3D_indexing[localised_frame_neighbours_points3D_ids[i]]
        visibility_matrix[:, vm_point3D_index] = (0.5) ** (t / t1_2) * visibility_matrix[:, vm_point3D_index]

sum_over_columns = visibility_matrix.sum(axis=0)
np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/sum_over_columns_new.txt", sum_over_columns)
np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/visibility_matrix_new.txt", visibility_matrix)