import numpy as np
from tensorflow import keras
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import colmap
from database import COLMAPDatabase
from query_image import get_image_id, get_keypoints_xy, get_queryDescriptors

db_path = sys.argv[1]
images_dir = sys.argv[2]
image_list_file = sys.argv[3]
model_path = sys.argv[4]
save_path = sys.argv[5]

# make sure the templates_ini/feature_extractions file are the same between Mobile-Pose.. and fullpipeline
colmap.feature_extractor(db_path, images_dir, image_list_file, query=True)
db = COLMAPDatabase.connect(db_path)

with open(image_list_file) as f:
    query_images = f.readlines()
query_images = [x.strip() for x in query_images]

model = keras.models.load_model(model_path)

for i in range(len(query_images)):
    q_img = query_images[i]
    image_id = get_image_id(db, q_img)
    # keypoints data
    keypoints_xy = get_keypoints_xy(db, image_id)
    queryDescriptors = get_queryDescriptors(db, image_id)

    predictions = model.predict(queryDescriptors)

    data = np.concatenate((keypoints_xy, predictions), axis=1)
    data = data[data[:, 2].argsort()[::-1]]

    np.savetxt(save_path + q_img.split(".")[0]+".txt", data)