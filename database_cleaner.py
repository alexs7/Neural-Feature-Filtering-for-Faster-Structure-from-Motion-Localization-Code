from query_image import read_images_binary
from database import COLMAPDatabase

#localised query images
query_images_file = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_name.txt"
with open(query_images_file) as f:
    query_images = f.readlines()
query_images = [x.strip() for x in query_images]

db = COLMAPDatabase.connect("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/all_models/base_model/database.db")
image_names = db.execute("SELECT name FROM images")
image_names = image_names.fetchall()

cursor = db.cursor()
for i in range(len(query_images)):
    print("Removing image " + str(i) + "/" + str(len(query_images)), end="\r")

    image_to_remove_name = query_images[i]
    image_id = db.execute("SELECT image_id FROM images WHERE name = " + "'" + image_to_remove_name + "'")
    image_id = str(image_id.fetchone()[0])

    cursor.execute("DELETE FROM images WHERE image_id = ?", (image_id,))
    db.commit()

    cursor.execute("DELETE FROM descriptors WHERE image_id = ?", (image_id,))
    db.commit()

    cursor.execute("DELETE FROM keypoints WHERE image_id = ?", (image_id,))
    db.commit()

print("\n Done")