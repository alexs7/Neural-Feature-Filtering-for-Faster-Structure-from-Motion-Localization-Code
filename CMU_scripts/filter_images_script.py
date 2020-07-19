# This file is to be used to separate files in folders (per weather condition) from the query folder. Only to use with CMU-Seasons dataset
# The ones in database folder are used for reconstruction
#  Note: rm session_1/* session_2/* session_3/* session_4/* session_5/* session_6/* session_7/* session_8/* session_9/* -> removed all seperated files if needed
import glob
import os
import subprocess
import sys
from datetime import datetime

Sunny_No_Foliage = [] # 4 Apr 2011
Sunny_Foliage = [] # 1 Sep 2010, 15 Sep 2010, 19 Oct 2010
Cloudy_Foliage = [] # 1 Oct 2010
Overcast_Mixed_Foliage = [] # 28 Oct 2010
Low_Sun_Mixed_Foliage = [] # 3 Nov 2010, 12 Nov 2010
Cloudy_Mixed_Foliage = [] # 22 Nov 2010
Low_Sun_No_Foliage_Snow = [] # 21 Dec 2010
Low_Sun_Foliage = [] # 4 Mar 2011
Overcast_Foliage = [] # 28 Jul 2011

i = 0
# example: "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/official_datasets/CMU-Seasons-Extended/slice2/query/"
base_dir = sys.argv[1]
os.chdir(base_dir)
dst_folder = "filtered/" #has to be added manually, along with all the other folders in it
for file in glob.glob("*.jpg"):
    if(file.split('_')[2] == 'c0'):
        i += 1

        # for database images
        # subprocess.call(["cp", base_dir + file, base_dir + dst_folder])

        # for query images
        if(1):
            timestamp = int(file.split('_')[3].split('us')[0])
            dt = datetime.fromtimestamp(timestamp/1000000)
            day = dt.day
            month = dt.month

            if(day == 4 and month == 4):
                Sunny_No_Foliage.append(file)
                subprocess.call(["cp", base_dir + file, base_dir + dst_folder + "session_1/"]) #this should be empty..
            if((day == 1 and month == 9) or (day == 15 and month == 9) or (day == 19 and month == 10)):
                subprocess.call(["cp", base_dir + file, base_dir + dst_folder + "session_2/"])
                Sunny_Foliage.append(file)
            if(day == 1 and month == 10):
                subprocess.call(["cp", base_dir + file, base_dir + dst_folder + "session_3/"])
                Cloudy_Foliage.append(file)
            if(day == 26 and month == 10): #this should be 28/10 but I think they made a mistake it is 26/10
                subprocess.call(["cp", base_dir + file, base_dir + dst_folder + "session_4/"])
                Overcast_Mixed_Foliage.append(file)
            if((day == 3 and month == 11) or (day == 12 and month == 11)):
                subprocess.call(["cp", base_dir + file, base_dir + dst_folder + "session_5/"])
                Low_Sun_Mixed_Foliage.append(file)
            if(day == 22 and month == 11):
                subprocess.call(["cp", base_dir + file, base_dir + dst_folder + "session_6/"])
                Cloudy_Mixed_Foliage.append(file)
            if(day == 21 and month == 12):
                subprocess.call(["cp", base_dir + file, base_dir + dst_folder + "session_7/"])
                Low_Sun_No_Foliage_Snow.append(file)
            if(day == 4 and month == 3):
                subprocess.call(["cp", base_dir + file, base_dir + dst_folder + "session_8/"])
                Low_Sun_Foliage.append(file)
            if (day == 28 and month == 7):
                subprocess.call(["cp", base_dir + file, base_dir + dst_folder + "session_9/"])
                Overcast_Foliage.append(file)

print("Sizes:")
print("Total: " + str(i))
print("Sunny_No_Foliage: " +str(len(Sunny_No_Foliage)))
print("Sunny_Foliage: " +str(len(Sunny_Foliage)))
print("Cloudy_Foliage: " +str(len(Cloudy_Foliage)))
print("Overcast_Mixed_Foliage: " +str(len(Overcast_Mixed_Foliage)))
print("Low_Sun_Mixed_Foliage: " +str(len(Low_Sun_Mixed_Foliage)))
print("Cloudy_Mixed_Foliage: " +str(len(Cloudy_Mixed_Foliage)))
print("Low_Sun_No_Foliage_Snow: " +str(len(Low_Sun_No_Foliage_Snow)))
print("Low_Sun_Foliage: " +str(len(Low_Sun_Foliage)))
print("Overcast_Foliage: " +str(len(Overcast_Foliage)))
