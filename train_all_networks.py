import os
import sys
import subprocess

# This file is used to train all the required networks
# it takes as a parameter a directory where the dataset is located (CMU/slice3, Coop/slice1 etc etc)
# before this you will need to run get_points_3D_mean_desc_single_model_ml.py

# Example command:
# maybe disown the proccess when running it ?
# python3 train_all_networks.py colmap_data/CMU_data/slice3/ Extended_CMU_slice3 & disown

data_path = sys.argv[1] #for example "colmap_data/CMU_data/slice3/"
name_of_model = sys.argv[2] #for example "Extended_CMU_slice3" , name of network architecture and dataset used

# Run baseline random features and baseline all features (system commands)
combined_model_command = "python3 combined_models_4.py "+data_path+" 32768 1000 " + name_of_model
classification_model_command = "python3 classification_4.py "+data_path+" 32768 1000 " + name_of_model
# on only matched features to be used only with classifier
regression_model_command_score_per_image = "python3 regression_4.py "+data_path+" 32768 1000 " + name_of_model + " 1 " + "score_per_image"
regression_model_command_score_per_session = "python3 regression_4.py "+data_path+" 32768 1000 " + name_of_model + " 1 " + "score_per_session"
regression_model_command_score_visibility = "python3 regression_4.py "+data_path+" 32768 1000 " + name_of_model + " 1 " + "score_visibility"
# on all features to be used on it's own
regression_on_all_model_command_score_per_image = "python3 regression_4.py "+data_path+" 32768 1000 All" + name_of_model + " 0 " + "score_per_image"
regression_on_all_model_command_score_per_session = "python3 regression_4.py "+data_path+" 32768 1000 All" + name_of_model + " 0 " + "score_per_session"
regression_on_all_model_command_score_visibility = "python3 regression_4.py "+data_path+" 32768 1000 All" + name_of_model + " 0 " + "score_visibility"

print("Commands to run:")
print(combined_model_command) #TODO: add the other two cases ? score_session ,and score_visibility?
print(classification_model_command)
print(regression_model_command_score_per_image)
print(regression_model_command_score_per_session)
print(regression_model_command_score_visibility)
print(regression_on_all_model_command_score_per_image)
print(regression_on_all_model_command_score_per_session)
print(regression_on_all_model_command_score_visibility)

print("Training started..")
os.system(combined_model_command)
os.system(classification_model_command)
os.system(regression_model_command_score_per_image)
os.system(regression_model_command_score_per_session)
os.system(regression_model_command_score_visibility)
os.system(regression_on_all_model_command_score_per_image)
os.system(regression_on_all_model_command_score_per_session)
os.system(regression_on_all_model_command_score_visibility)
print("Training done!")




