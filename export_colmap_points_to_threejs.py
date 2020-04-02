from point3D_loader import read_points3d_default

path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/26_03_2020/fresh_uni_large/model_run_1/model/0/points3D.bin"
file_output = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/26_03_2020/fresh_uni_large/model_run_1/model/0/points3D_threeJS.txt"
points3D = read_points3d_default(path)

f = open(file_output, 'a+')
for k in points3D:
    line = str(points3D[k].id) + "," + str(points3D[k].xyz[0]) + "," + str(points3D[k].xyz[1]) + "," + str(points3D[k].xyz[2]) + \
           "," + str(points3D[k].rgb[0]) + "," + str(points3D[k].rgb[1]) + "," + str(points3D[k].rgb[2])
    f.write(line + "\n")

f.close()