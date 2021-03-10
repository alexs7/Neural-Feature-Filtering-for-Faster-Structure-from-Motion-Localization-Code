import cv2
import numpy as np
import sys

frame_code = sys.argv[1]
percentage = sys.argv[2]

# thse are already sorted
keypoints_data = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/coop_local/ML_data/results/frames_ml_data/frame_"+frame_code+".txt")
image_original = cv2.imread("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/coop_local/ML_data/results/test_images/2020-06-22/frame_"+frame_code+".jpg")
image_reference = cv2.imread("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/coop_local/ML_data/results/test_images/2020-06-22/frame_"+frame_code+".jpg")

data_len = keypoints_data.shape[0]
print("Data length: " + str(data_len))
top_points_len = int(data_len * float(percentage) / 100)
print("Top points length: " + str(top_points_len))
red = (0, 0, 255)
blue = (255, 0, 0)

for i in range(top_points_len):
    x = int(keypoints_data[i][0])
    y = int(keypoints_data[i][1])
    center = (x, y)
    cv2.circle(image_original, center, 4, red, -1)
cv2.imwrite("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/coop_local/ML_data/results/local_debug_data_for_threejs/"+"frame_"+frame_code+"_ml.jpg", image_original)

np.random.shuffle(keypoints_data)
for i in range(top_points_len):
    x = int(keypoints_data[i][0])
    y = int(keypoints_data[i][1])
    center = (x, y)
    cv2.circle(image_reference, center, 4, red, -1)
cv2.imwrite("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/coop_local/ML_data/results/local_debug_data_for_threejs/"+"frame_"+frame_code+"_random.jpg", image_reference)

# def show_projected_points(image_path, sorted_matches, sample, output):
#     np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/sorted_matches.npy", sorted_matches)
#     lowes_distances = sorted_matches[:, 6] / sorted_matches[:, 6].sum()
#     heatmap_vals = sorted_matches[:, 7] / sorted_matches[:, 7].sum()
#     points2D_all = sorted_matches[:, 0:2]
#     points2D_sample = sample[:, 0:2]
#     red = (0, 0, 255)
#     blue = (255, 0, 0)
#     image = cv2.imread(image_path)
#     for i in range(int(len(points2D_all))):
#         x = int(points2D_all[i][0])
#         y = int(points2D_all[i][1])
#         center = (x, y)
#         cv2.circle(image, center, 2, blue, -1)
#     #     also add text
#         val = lowes_distances[i] * heatmap_vals[i]
#         # cv2.putText(image, "{:.10f}".format(val), (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255, 255, 255), thickness = 2 )
#     for i in range(len(points2D_sample)):
#         val = lowes_distances[i] * heatmap_vals[i]
#         # print("sample's val :" + "{:.10f}".format(val))
#         # print("sample's point3D id :" + str(sample[i,5]))
#         x = int(points2D_sample[i][0])
#         y = int(points2D_sample[i][1])
#         # print("sample's x and y :" + str(x) + " " + str(y))
#         center = (x, y)
#         cv2.circle(image, center, 5, red, -1)
#     cv2.imwrite("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/"+output, image)


