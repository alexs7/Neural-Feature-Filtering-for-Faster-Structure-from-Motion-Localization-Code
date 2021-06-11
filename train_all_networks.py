import os
import sys
import subprocess

# This file is used to train all the required networks
# it takes as a parameter a directory where the dataset is located
# before this you will need to run get_points_3D_mean_desc_single_model_ml.py

# Example command:
# maybe disown the proccess when running it ?
# python3 train_all_networks.py colmap_data/CMU_data/slice3/ Extended_CMU_slice3

data_path = sys.argv[1] #for example "colmap_data/CMU_data/slice3/"
name_of_model = sys.argv[2] #for example "Extended_CMU_slice3" , name of network architecture and dataset used

# Run baseline random features and baseline all features (system commands)
combined_model_command = "python3 combined_4.py "+data_path+" 32768 1000 " + name_of_model
classification_model_command = "python3 classification_4.py "+data_path+" 32768 1000 " + name_of_model
regression_model_command = "python3 regression_4.py "+data_path+" 32768 1000 " + name_of_model
regression_on_all_model_command = "python3 regression_4.py "+data_path+" 32768 1000 " + name_of_model + " 0"

print("Commands to run:")
print(combined_model_command)
print(classification_model_command)
print(regression_model_command)
print(regression_model_command)

print("Training started..")
subprocess.run(combined_model_command)
subprocess.run(classification_model_command)
subprocess.run(regression_model_command)
subprocess.run(regression_on_all_model_command)
print("Training done!")




