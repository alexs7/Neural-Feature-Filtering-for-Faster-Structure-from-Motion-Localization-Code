from point3D_loader import read_points3d_default
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

points3D = read_points3d_default("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/points3D.bin")

class Point3D:
  def __init__(self, id, index, xyz, seen_by_no):
    self.id = id
    self.index = index
    self.xyz = xyz
    self.seen_by_no = seen_by_no

print("Getting observation count from all traversals")
observations = []
for k,v in points3D.items():
    observations.append(len(np.unique(v.image_ids)))
observations = np.array(observations)
observations_mean = observations.mean()

# create an index and point_id relationship and points3D objects
point3D_index = 0
points3D_objects = []
for k, v in points3D.items():
    p = Point3D(k,point3D_index,v.xyz, len(np.unique(v.image_ids)))
    points3D_objects.append(p)
    point3D_index = point3D_index + 1

# data from matlab
v_matrix = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/visibility_matrix_new.txt")
col_sum = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/col_sum.txt")

# write different outputs based on custom criteria
# 1 - mean maybe ?
mean = np.mean(col_sum)
col_sum_over_mean = np.where(col_sum > mean, col_sum, 0)

xyz_to_render = np.empty([0, 3])
for i in range(len(col_sum_over_mean)):
    if(col_sum_over_mean[i] != 0):
        for p in points3D_objects:
            if(p.index == i):
                xyz_to_render = np.r_[xyz_to_render, np.reshape(p.xyz, [1,3])]

no_new_points = len(xyz_to_render)
print("Reduced to " + str(no_new_points * 100 / len(col_sum)) + "%")
np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/col_sum_points_mean.txt", xyz_to_render)

# 2 - mean and by how many cameras it is seen?
mean = np.mean(col_sum)
col_sum_over_mean = np.where(col_sum > mean, col_sum, 0)

xyz_to_render = np.empty([0, 3])
for i in range(len(col_sum_over_mean)):
    if(col_sum_over_mean[i] != 0):
        for p in points3D_objects:
            if(p.index == i and p.seen_by_no > observations_mean):
                xyz_to_render = np.r_[xyz_to_render, np.reshape(p.xyz, [1,3])]

no_new_points = len(xyz_to_render)
print("Reduced to " + str(no_new_points * 100 / len(col_sum)) + "%")
np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/col_sum_points_camera_and_sum_mean.txt", xyz_to_render)

# 3 - set cover ? (double check if you applied the algorithm correctly)
selected_3D_points = []
can_select = True

while can_select:
    max_col_index = np.argmax(col_sum)
    if()
    selected_3D_points.append()




breakpoint()

xyz_to_render = np.empty([0, 3])
for i in range(len(col_sum_over_mean)):
    if(col_sum_over_mean[i] != 0):
        for p in points3D_objects:
            if(p.index == i and p.seen_by_no > observations_mean):
                xyz_to_render = np.r_[xyz_to_render, np.reshape(p.xyz, [1,3])]