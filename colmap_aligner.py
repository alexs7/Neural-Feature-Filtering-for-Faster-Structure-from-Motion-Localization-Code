from query_image import get_image_camera_center
import colmap

# the model you want to align
path_to_model = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/common_models/generic/model_1/27_04_2020/coop_local/model/model/0"
path_to_geo_registered_model = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/common_models/generic/geo_registered_model"
path_to_text_file = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/common_models/generic/images.txt"
path_to_query_images_file = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_name.txt"
# the model that you are aligning to
path_to_images_model = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/images.bin"

with open(path_to_query_images_file) as f:
    query_images = f.readlines()
query_images = [x.strip() for x in query_images]

non_localised_frames = 0
localised_frames = 0
f = open(path_to_text_file, 'w')
for i in range(len(query_images)):
    camera_center = get_image_camera_center(path_to_images_model, query_images[i])
    if(camera_center.size != 0):
        localised_frames = localised_frames + 1
        f.write(query_images[i] + " " + str(camera_center[0]) + " " + str(camera_center[1]) + " " + str(camera_center[2]) + "\n")
    else:
        non_localised_frames = non_localised_frames + 1
f.close()

breakpoint()
print("Result " + str(localised_frames*100/len(query_images)))

colmap.model_aligner(path_to_model,path_to_geo_registered_model,path_to_text_file)
