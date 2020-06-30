import glob, os
import subprocess

base_dir = "/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/all_data_and_models/coop_local/2020-06-22/evening/run_1/only_jpgs/"

os.chdir(base_dir)
files = []
nth = 3

for file in glob.glob("*.jpg"):
    print(file)
    files.append(file)

sub_set = files[0::nth]

for file in sub_set:
    subprocess.call(["cp", base_dir+file,
                            base_dir+"/sub_set/"+file])