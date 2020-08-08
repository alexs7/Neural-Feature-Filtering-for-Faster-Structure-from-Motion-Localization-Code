import sys
import numpy as np
from database import COLMAPDatabase

db_path = sys.argv[1]

db = COLMAPDatabase.connect(db_path)
all_intrinsics = db.execute("SELECT params FROM cameras")
all_intrinsics = all_intrinsics.fetchall()

for intrinsics in all_intrinsics:
    data = COLMAPDatabase.blob_to_array(intrinsics[0], np.float64)
    print(data)
