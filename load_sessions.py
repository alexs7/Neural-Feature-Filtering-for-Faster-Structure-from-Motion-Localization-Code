# this file loads and writes the sessions' images in
# seperate files
from query_image import read_images_binary
import numpy as np

all_sessions_dic = {}

images_from_29_03 = read_images_binary("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/2020-03-29/coop_local/model/model/0/images.bin")
images_from_04_04 = read_images_binary("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/2020-04-04/coop_local/model/model/0/images.bin")
images_from_09_04 = read_images_binary("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/2020-04-09/coop_local_small/model/model/0/images.bin")
images_from_23_04 = read_images_binary("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/2020-04-23/coop_local_small/model/model/0/images.bin")
images_from_25_04 = read_images_binary("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/2020-04-25/coop_local/model/model/0/images.bin")
images_from_26_04 = read_images_binary("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/2020-04-26/coop_local/model/model/0/images.bin")
images_from_27_04 = read_images_binary("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/2020-04-27/coop_local/model/model/0/images.bin")
images_from_02_05 = read_images_binary("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/2020-05-02/coop_local/model/model/0/images.bin")
images_from_06_05 = read_images_binary("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/2020-05-06/coop_local/model/model/0/images.bin")

print("Getting the future sessions images' names..")
images_from_29_03_names = []
for k, v in images_from_29_03.items():
    images_from_29_03_names.append(v.name)

with open('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/sessions_data/images_names/images_from_29_03_names.txt', 'w') as f:
    for item in images_from_29_03_names:
        f.write("%s\n" % item)

all_sessions_dic['images_from_29_03'] = images_from_29_03_names

images_from_04_04_names = []
for k, v in images_from_04_04.items():
    images_from_04_04_names.append(v.name)

with open('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/sessions_data/images_names/images_from_04_04_names.txt', 'w') as f:
    for item in images_from_04_04_names:
        f.write("%s\n" % item)

all_sessions_dic['images_from_04_04'] = images_from_04_04_names

images_from_09_04_names = []
for k, v in images_from_09_04.items():
    images_from_09_04_names.append(v.name)

with open('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/sessions_data/images_names/images_from_09_04_names.txt', 'w') as f:
    for item in images_from_09_04_names:
        f.write("%s\n" % item)

all_sessions_dic['images_from_09_04'] = images_from_09_04_names

images_from_23_04_names = []
for k, v in images_from_23_04.items():
    images_from_23_04_names.append(v.name)

with open('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/sessions_data/images_names/images_from_23_04_names.txt', 'w') as f:
    for item in images_from_23_04_names:
        f.write("%s\n" % item)

all_sessions_dic['images_from_23_04'] = images_from_23_04_names

images_from_25_04_names = []
for k, v in images_from_25_04.items():
    images_from_25_04_names.append(v.name)

with open('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/sessions_data/images_names/images_from_25_04_names.txt', 'w') as f:
    for item in images_from_25_04_names:
        f.write("%s\n" % item)

all_sessions_dic['images_from_25_04'] = images_from_25_04_names

images_from_26_04_names = []
for k, v in images_from_26_04.items():
    images_from_26_04_names.append(v.name)

with open('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/sessions_data/images_names/images_from_26_04_names.txt', 'w') as f:
    for item in images_from_26_04_names:
        f.write("%s\n" % item)

all_sessions_dic['images_from_26_04'] = images_from_26_04_names

images_from_27_04_names = []
for k, v in images_from_27_04.items():
    images_from_27_04_names.append(v.name)

with open('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/sessions_data/images_names/images_from_27_04_names.txt', 'w') as f:
    for item in images_from_27_04_names:
        f.write("%s\n" % item)

all_sessions_dic['images_from_27_04'] = images_from_27_04_names

images_from_02_05_names = []
for k, v in images_from_02_05.items():
    images_from_02_05_names.append(v.name)

with open('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/sessions_data/images_names/images_from_02_05_names.txt', 'w') as f:
    for item in images_from_02_05_names:
        f.write("%s\n" % item)

all_sessions_dic['images_from_02_05'] = images_from_02_05_names

images_from_06_05_names = []
for k, v in images_from_06_05.items():
    images_from_06_05_names.append(v.name)

with open('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/sessions_data/images_names/images_from_06_05_names.txt', 'w') as f:
    for item in images_from_06_05_names:
        f.write("%s\n" % item)

all_sessions_dic['images_from_06_05'] = images_from_06_05_names

# save dictionary
np.save('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/sessions_data/images_names/all_sessions_dic.npy', all_sessions_dic)