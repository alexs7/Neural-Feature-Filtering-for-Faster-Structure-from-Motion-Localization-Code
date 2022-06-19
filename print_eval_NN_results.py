# example command: python3 print_eval_NN_results.py 5
# you will run this after, model_evaluator.py
# Note that for the 5% case many images did NOT localise, so there is not pose
# This script will generate a .txt that you copy in Excel, then export to .csv and run plots/plots.py with the .csv

# you have to move the resulting file from weatherwax to the alienware machine then to your laptop
# use "plots/results_temp.xls" to copy paste values from the .csv produced here then export to csv
import os
import sys

percent = sys.argv[1]
file = "NN_results_excel_"+str(percent)+".txt"

if(os.path.isfile(file)):
    print("Removing old file..")
    os.remove(file)

cmu_slice3 = "colmap_data/CMU_data/slice3/ML_data/results_evaluator_"+str(percent)+".csv"
cmu_slice4 = "colmap_data/CMU_data/slice4/ML_data/results_evaluator_"+str(percent)+".csv"
cmu_slice6 = "colmap_data/CMU_data/slice6/ML_data/results_evaluator_"+str(percent)+".csv"
cmu_slice10 = "colmap_data/CMU_data/slice10/ML_data/results_evaluator_"+str(percent)+".csv"
cmu_slice11 = "colmap_data/CMU_data/slice11/ML_data/results_evaluator_"+str(percent)+".csv"
coop_slice1 = "colmap_data/Coop_data/slice1/ML_data/results_evaluator_"+str(percent)+".csv"

results  = [cmu_slice3, cmu_slice4, cmu_slice6, cmu_slice10, cmu_slice11, coop_slice1]

for result in results:
    os.system("cat " + result + " >> " + " " + file)
    os.system("echo " + "" + " >> " + " " + file)
    os.system("echo " + "" + " >> " + " " + file)