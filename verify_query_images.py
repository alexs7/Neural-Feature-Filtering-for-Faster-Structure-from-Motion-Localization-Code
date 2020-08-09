
import glob
import os
import sys
from point3D_loader import read_points3d_binary, read_points3d_binary_id
from query_image import read_images_binary, get_image_by_name, get_intrinsics_from_camera_bin, \
    get_query_image_pose_from_images, save_image_projected_points

base_path = sys.argv[1] # i.e /home/alex/fullpipeline/colmap_data/CMU_data/slice3/

images = read_images_binary(base_path+"gt/model/images.bin")
query_images_path = base_path+"gt/images/session_query/"
K = get_intrinsics_from_camera_bin(base_path+"gt/model/cameras.bin", 3)

os.chdir(query_images_path)
for file in glob.glob("*.jpg"):
    name = "session_query/"+file
    output_path = base_path+"gt/images/projected/"+file
    image = get_image_by_name(name, images)
    points3D_image = read_points3d_binary_id(base_path+"gt/model/points3D.bin", image.id)
    query_pose = get_query_image_pose_from_images(name, images)
    save_image_projected_points(query_images_path+file,K,query_pose,points3D_image,output_path)



