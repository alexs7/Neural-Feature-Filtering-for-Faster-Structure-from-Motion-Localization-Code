# this is to get a huge np matrix with each points mean desc
# be careful that you can get the base model's avg descs or the complete's model descs
import sqlite3
import numpy as np
import sys
from point3D_loader import read_points3d_default
from query_image import read_images_binary, image_localised

IS_PYTHON3 = sys.version_info[0] >= 3

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

db = COLMAPDatabase.connect("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/database.db")

# by "complete model" I mean all the frames from future sessions localised in the base model (28/03)
complete_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/images.bin"
complete_model_all_images = read_images_binary(complete_model_images_path)
complete_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/points3D.bin"
points3D = read_points3d_default(complete_model_points3D_path) # base model's 3D points

# create points id and index relationship
point3D_index = 0
points3D_indexing = {}
for key, value in points3D.items():
    points3D_indexing[point3D_index] = value.id
    point3D_index = point3D_index + 1

# get base images
base_images_names = []
with open("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/base_images.txt") as f:
    base_images_names = f.readlines()
base_images_names = [x.strip() for x in base_images_names]

base_images_ids = []
for name in base_images_names:
    id = image_localised(name, complete_model_all_images)
    base_images_ids.append(id)

print("base_images_ids size " + str(len(base_images_ids)))

# match against the base model - getting all 3D points avg desc and save them in a huge file
points_id_desc = {}
points_mean_descs = np.empty([0, 128])
for i in range(0,len(points3D)):
    print("Doing point " + str(i) + "/" + str(len(points3D)) + " with id " +str(points3D_indexing[i]))
    point_id = points3D_indexing[i]
    points3D_descs = np.empty([0, 128])
    for k in range(len(points3D[point_id].image_ids)): #unique here doesn't really matter
        id = points3D[point_id].image_ids[k]
        if(id in base_images_ids): # TODO: remove this to get all the points average
            data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(id) + "'")
            data = blob_to_array(data.fetchone()[0], np.uint8)
            descs_rows = int(np.shape(data)[0] / 128)
            descs = data.reshape([descs_rows, 128])
            desc = descs[points3D[point_id].point2D_idxs[k]]
            desc = desc.reshape(1, 128)
            points3D_descs = np.r_[points3D_descs, desc]
    points_mean_descs = np.r_[points_mean_descs,points3D_descs.mean(axis=0).reshape(1,128)]

np.savetxt('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/benchmarks/points_mean_descs_base.txt', points_mean_descs)