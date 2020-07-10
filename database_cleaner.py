from query_image import read_images_binary
from database import COLMAPDatabase

#localised query images
images_to_delete_from_db_file = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/current_query_image/session_images.txt"
with open(images_to_delete_from_db_file) as f:
    images_to_delete_from_db = f.readlines()
images_to_delete_from_db = [x.strip() for x in images_to_delete_from_db]

db = COLMAPDatabase.connect("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/all_models/live_model/database_base.db")
image_names = db.execute("SELECT name FROM images")
image_names = image_names.fetchall()

cursor = db.cursor()
for i in range(len(images_to_delete_from_db)):
    print("Removing image " + str(i+1) + "/" + str(len(images_to_delete_from_db)), end="\r")

    image_to_remove_name = images_to_delete_from_db[i]
    image_id = db.execute("SELECT image_id FROM images WHERE name = " + "'" + image_to_remove_name + "'")
    image_id = str(image_id.fetchone()[0])

    cursor.execute("DELETE FROM images WHERE image_id = ?", (image_id,))
    db.commit()

    cursor.execute("DELETE FROM descriptors WHERE image_id = ?", (image_id,))
    db.commit()

    cursor.execute("DELETE FROM keypoints WHERE image_id = ?", (image_id,))
    db.commit()

print("\n Done")