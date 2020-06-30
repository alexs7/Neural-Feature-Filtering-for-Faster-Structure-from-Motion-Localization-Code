import numpy as np

from export_colmap_data_to_threejs import savePoints3DxyzToFile
from point3D_loader import read_points3d_default, index_dict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

features_no = "1k"
exponential_decay_value = "0.5"

complete_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/" + features_no + "/points3D.bin"
points3D = read_points3d_default(complete_model_points3D_path)
points3D_indexing = index_dict(points3D)

rgb_avg = []
for k,v in points3D.items():
    rgb_avg.append(v.rgb.mean())

rgb_avg = np.array(rgb_avg)
rgb_avg = rgb_avg.reshape([rgb_avg.shape[0] , 1])

heatmap_matrix = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/"+features_no+"/heatmap_matrix_" + str(exponential_decay_value) + ".txt")
points3D_sum_heatmap_vals = heatmap_matrix.sum(axis=0) #axis = 0, apply summation along rows axis of array, i.e sum columns
points3D_sum_heatmap_vals = points3D_sum_heatmap_vals.reshape([points3D_sum_heatmap_vals.shape[0] , 1])

data  = np.concatenate([points3D_sum_heatmap_vals, rgb_avg], axis=1)

print("Fitting Data")
kmeans = KMeans(n_clusters=10)
kmeans.fit(data)
y_kmeans = kmeans.predict(data)
centers = kmeans.cluster_centers_

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Exponential Decay Values vs RGB mean value (per points). No K-means / K-means Clusters: ' + str(kmeans.n_clusters), fontsize=10)

ax1.scatter(data[:, 0], data[:, 1], s=0.4)
ax1.set_xlabel('Exponential Decay Value (the higher the more static the point)', fontsize=9)
ax1.set_ylabel('Mean RGB value of 3D points', fontsize=9)

ax2.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=0.4, cmap='jet')
ax2.scatter(centers[:, 0], centers[:, 1], c='red', s=32, alpha=0.7)
ax2.set_xlabel('Exponential Decay Value (the higher the more static the point)', fontsize=9)
ax2.set_ylabel('Mean RGB value of 3D points', fontsize=9)

plt.show()

breakpoint()

max_x_value_index = centers.argmax(axis=0)[0]
points3D_max = np.where(y_kmeans == max_x_value_index)[0]

points3D_xyz = []
for i in range(len(points3D_max)):
    points3D_id = points3D_indexing[points3D_max[i]]
    points3D_xyz.append(points3D[points3D_id].xyz)

savePoints3DxyzToFile(points3D_xyz)


