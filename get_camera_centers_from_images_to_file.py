import sys
from query_image import read_images_binary, get_images_camera_centers, get_images_names_bin, get_images_camera_quats, get_intrinsics_from_camera_bin, \
    get_images_camera_principal_axis_vectors

images_path = sys.argv[1]
cameras_path = sys.argv[2]
path_to_text_file = sys.argv[3]

names = get_images_names_bin(images_path)
images = read_images_binary(images_path)
K = get_intrinsics_from_camera_bin(cameras_path, 1)
centers = get_images_camera_centers(images)
quats = get_images_camera_quats(images)
principal_axis_vectors = get_images_camera_principal_axis_vectors(images, Ks)

f = open(path_to_text_file, 'w')
for name in names:
    data = str(centers[name][0]) + " " + str(centers[name][1]) + " " + str(centers[name][2]) + " " + str(quats[name][0]) + " " + str(quats[name][1]) + " " \
           + str(quats[name][2]) + " " + str(quats[name][3]) + " " + str(principal_axis_vectors[name][0]) + " " + str(principal_axis_vectors[name][1]) \
           + " " + str(principal_axis_vectors[name][2]) + " " + name + "\n"
    f.write(data)
f.close()