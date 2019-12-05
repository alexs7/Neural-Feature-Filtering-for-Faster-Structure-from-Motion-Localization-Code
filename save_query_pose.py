import sys
from colmap_database import COLMAPDatabase

db_path = sys.argv[1]
txt_images_file = sys.argv[2]

db = COLMAPDatabase.connect(db_path)

f = open(txt_images_file, 'r')
lines = f.readlines()
lines = lines[4:]
pose_data = lines[-2].split(" ")
f.close()

qw = pose_data[1]
qx = pose_data[2]
qy = pose_data[3]
qz = pose_data[4]
tx = pose_data[5]
ty = pose_data[6]
tz = pose_data[7]

db.execute("UPDATE images set prior_qw = "+qw+", prior_qx = "+qx+", prior_qy = "+qy+", prior_qz = "+qz+", prior_tx = "+tx+", prior_ty = "+ty+", prior_tz = "+tz+" WHERE name = "+"'"+"query.jpg"+"'")
db.commit()

