import glob, os
import subprocess
import sys

base_dir = sys.argv[1]

os.chdir(base_dir)
files = []
nth = 2

for file in glob.glob("*.jpg"):
    files.append(file)

sub_set = files[0::nth]

files_no = 0
for file in sub_set:
    subprocess.call(["cp", base_dir + "/"+file,
                            base_dir+"/sub_set/"+file])
    files_no += 1

print("Files no: " + str(files_no))