# This is to split the frame into equal parts and measure how many points are in there - variance - TODO: WIP
from query_image import read_images_binary
from show_2D_points import show_projected_points
import numpy as np

path_images_new_model = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/images.bin"
images_new = read_images_binary(path_images_new_model)

for k,v in images_new.items():
    if(v.name == "frame_1585500887093.jpg"):
        show_projected_points("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/current_query_image/frame_1585500887093.jpg", v.xys, (0, 255, 0), "detected_keypoints.jpg")
        points2D = np.empty([0,2])
        for i in range(len(v.point3D_ids)):
            if(v.point3D_ids[i] != -1):
                points2D = np.r_[points2D, v.xys[i].reshape([1, 2])]

        print("Visible 3D points no: " + str(len(points2D)))
        show_projected_points("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/detected_keypoints.jpg", points2D, (0, 0, 255), "projected_3D_points.jpg")
        break