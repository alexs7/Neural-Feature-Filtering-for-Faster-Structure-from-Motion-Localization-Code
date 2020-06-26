import cv2
import numpy as np

# K = np.loadtxt("matrices/pixel_intrinsics_low_640.txt")

def show_projected_points(image_path, points2D, color, output):
    image = cv2.imread(image_path)
    for i in range(len(points2D)):
        x = int(points2D[i][0])
        y = int(points2D[i][1])
        center = (x, y)
        cv2.circle(image, center, 8, color, -1)
    # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/"+output, image)

# arcore_correspondences = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/query_data/cpuImageCorrespondences_1579029504279.txt")
# points2D = arcore_correspondences[:,0:2]
# image_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_data/frame_1579029504279.jpg"
#
# show_projected_points(image_path, points2D)

