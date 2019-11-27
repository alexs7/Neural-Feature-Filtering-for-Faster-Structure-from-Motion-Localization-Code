from os import listdir
from os.path import isfile, join
import pdb
import sys
import point_projector

images_path = sys.argv[1]
correspondences_path = sys.argv[2]

images = [image for image in listdir(images_path) if not image.startswith('.') and isfile(join(images_path, image))]
correspondences = [correspondence for correspondence in listdir(correspondences_path) if not correspondence.startswith('.') and isfile(join(correspondences_path, correspondence))]

images.sort()
correspondences.sort()

for i in range(len(images)):
    point_projector.project_points("colmap_data/data5/images/"+images[i], "colmap_data/data5/correspondences/"+correspondences[i], i)

pdb.set_trace()