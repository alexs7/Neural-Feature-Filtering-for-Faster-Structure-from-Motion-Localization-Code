import glob

for fname in sorted(glob.glob("colmap_data/data6/query_data/*.jpg")):
    print(fname)