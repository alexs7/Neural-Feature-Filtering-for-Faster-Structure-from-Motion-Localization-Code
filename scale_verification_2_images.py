import numpy as np
from query_image import read_images_binary, get_image_camera_center_by_name

images = read_images_binary('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/images.bin')
cnt1 = get_image_camera_center_by_name('frame_1592381233843.jpg',images)
cnt2 = get_image_camera_center_by_name('frame_1592381203802.jpg',images)
scale = 2.307638538053233823e-01
dist = scale * np.linalg.norm(cnt1 - cnt2)

print(dist)