from query_image import image_localised, read_images_binary
import numpy as np

base_images_names = []
with open("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/base_images.txt") as f:
    base_images_names = f.readlines()
base_images_names = [x.strip() for x in base_images_names]

query_images_names = []
with open("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_name.txt") as f:
    query_images_names = f.readlines()
query_images_names = [x.strip() for x in query_images_names]

complete_model_images = base_images_names + query_images_names

def get_images(features_no):

    images_localised = []
    images_not_localised = []
    complete_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/multiple_localised_models/" + features_no + "/images.bin"
    complete_model_all_images = read_images_binary(complete_model_images_path)

    for image in complete_model_images:
        image_id = image_localised(image, complete_model_all_images)
        if(image_id != None):
            images_localised.append(image)
        else:
            images_not_localised.append(image)

    print("images_localised : " + str(len(images_localised)))
    print("images_not_localised : " + str(len(images_not_localised)))

    # Needless to say they include the base images too
    np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/images_localised_and_not_localised/" + features_no +"/images_localised.txt",images_localised,fmt='%s')
    np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/images_localised_and_not_localised/" + features_no +"/images_not_localised.txt",images_not_localised,fmt='%s')

colmap_features_no = ["2k", "1k", "0.5k", "0.25k"]
# run for each no of features
for features_no in colmap_features_no:
    print("Getting localized and non localised images for features_no " + features_no )
    get_images(features_no)