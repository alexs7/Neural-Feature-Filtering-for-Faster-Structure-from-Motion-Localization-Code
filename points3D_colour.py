import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from export_colmap_data_to_threejs import savePoints3DxyzToFile
from point3D_loader import read_points3d_default, index_dict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

features_no = "1k"
exponential_decay_value = "0.5"

live_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/points3D.bin"
points3D = read_points3d_default(live_model_points3D_path)
points3D_indexing = index_dict(points3D)

sorted_matches = np.load("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/sorted_matches.npy")

xs = []
ys = []
sorting_vals = []
points3D_avg_color_vals = []
for i in range(len(sorted_matches)):
    x = sorted_matches[i,0]
    y = sorted_matches[i,1]
    val = sorted_matches[i, 6] * sorted_matches[i, 7]
    xs.append(x)
    ys.append(y)
    sorting_vals.append(val)
    points3D_index = sorted_matches[i, 5]
    point3D_id = points3D_indexing[points3D_index]
    current_point3D = points3D[point3D_id]
    breakpoint()


xs = np.array(xs)
xs = xs.reshape([xs.shape[0] , 1])

ys = np.array(ys)
ys = ys.reshape([ys.shape[0] , 1])

sorting_vals = np.array(sorting_vals)
sorting_vals = sorting_vals.reshape([sorting_vals.shape[0] , 1])

data  = np.concatenate([xs, ys, sorting_vals], axis=1)

print("Fitting Data")
kmeans = KMeans(n_clusters=4)
kmeans.fit(data)
y_kmeans = kmeans.predict(data)
centers = kmeans.cluster_centers_

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Pixel and Exponential Values in 3D: ' + str(kmeans.n_clusters), fontsize=10)

ax1=Axes3D(fig)
ax2=Axes3D(fig)

ax1.scatter(data[:, 0], data[:, 1], data[:, 2], s=0.4)
ax1.set_xlabel('Exponential Decay Value (the higher the more static the point)', fontsize=9)
ax1.set_ylabel('Mean RGB value of 3D points', fontsize=9)

point_closest_to_c1 = data[np.argmin(kmeans.transform(data)[:,0]),:]
point_closest_to_c2 = data[np.argmin(kmeans.transform(data)[:,1]),:]
point_closest_to_c3 = data[np.argmin(kmeans.transform(data)[:,2]),:]
point_closest_to_c4 = data[np.argmin(kmeans.transform(data)[:,3]),:]

ax2.scatter(data[:, 0], data[:, 1], data[:, 2], c=y_kmeans, s=20, cmap='jet')
ax2.scatter(point_closest_to_c1[0], point_closest_to_c1[1], point_closest_to_c1[2], c='red', s=120, alpha=0.9)
ax2.scatter(point_closest_to_c2[0], point_closest_to_c2[1], point_closest_to_c2[2], c='red', s=120, alpha=0.9)
ax2.scatter(point_closest_to_c3[0], point_closest_to_c3[1], point_closest_to_c3[2], c='red', s=120, alpha=0.9)
ax2.scatter(point_closest_to_c4[0], point_closest_to_c4[1], point_closest_to_c4[2], c='red', s=120, alpha=0.9)
ax2.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='green', s=60, alpha=0.7)

ax2.set_xlabel("X Values (Width)", fontsize=9)
ax2.set_ylabel("Y Values (Height)", fontsize=9)
ax2.set_zlabel('Exponential Decay Value (the higher the more static the point)', fontsize=9)

plt.show()


breakpoint()

max_x_value_index = centers.argmax(axis=0)[0]
points3D_max = np.where(y_kmeans == max_x_value_index)[0]

points3D_xyz = []
for i in range(len(points3D_max)):
    points3D_id = points3D_indexing[points3D_max[i]]
    points3D_xyz.append(points3D[points3D_id].xyz)

savePoints3DxyzToFile(points3D_xyz)


