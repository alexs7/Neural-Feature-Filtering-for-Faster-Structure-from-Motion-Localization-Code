import glob
import os
import sys

# /home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/live
# /home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/gt
dir = sys.argv[1]
os.chdir(dir)

session_nums = []
images = []
for folder in glob.glob("session_*"):
    i=0
    for file in glob.glob(folder+"/*.jpg"):
        image_name = file.split('/')[1]
        images.append(image_name)
        i+=1
    session_nums.append(i)

with open('../query_name.txt', 'w') as f:
    for image in images:
        f.write("%s\n" % image)

print("session_nums = " + str(session_nums) + " - Dont forget to add the base ones!")
