import numpy as np

def parse_points3D(points3D_text_file, query_image_id):

    f = open(points3D_text_file, 'r')
    lines = f.readlines()
    lines = lines[3:] #skip comments
    f.close()

    points3D = np.empty((0, 3))

    for i in range(len(lines)):
        line = lines[i].split()

        image_id_points_map = line[8:]

        print("Loading done " + str( round(i * 100 / len(lines)) ) + "%",  end = '\r' )

        for i in range(0, len(image_id_points_map) ,2):

            image_id = image_id_points_map[i]

            if(query_image_id == image_id):
                point_idx = image_id_points_map[i+1]
                points3D = np.append(points3D, [line[1], line[2], line[3]])

    points3D = points3D.reshape([int(len(points3D)/3),3])
    return points3D