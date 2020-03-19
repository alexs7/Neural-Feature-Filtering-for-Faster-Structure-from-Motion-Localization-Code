from point3D_loader import read_points3d_binary
import numpy as np

points3D = read_points3d_binary('/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/new_model/points3D.bin')
print("Number of COLMAP 3D Points: " + str(len(points3D)))
np.savetxt('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/all_points3D.txt', points3D)
