# TODO: Adapt for Predicting Matchability (2014) paper
# This file will load also the same files from print_eval_NN_results.py
# but with the Predicting Matchability (2014) results
# In this file I have added the percentage formatting

# example command: python3 print_eval_NN_results.py 5
# you will run this after, model_evaluator.py
# Note that for the 5% case many images did NOT localise, so there is not pose
# This script will generate a .txt that you copy in Excel, then export to .csv and run plots/plots.py with the .csv

# you have to move the resulting file from weatherwax to the alienware machine then to your laptop
# use "plots/results_temp.xls" to copy paste values from the .csv produced here then export to csv
import os
import sys
from plots.plots_variables import names_dict
import csv
import numpy as np

base_path = sys.argv[1]
percent = sys.argv[2] # NOTE: that I assume the I will be comparing to the 10% only
csv_output =  os.path.join(base_path, "results_new_"+str(percent)+".csv")

cmu_slice3 = "colmap_data/CMU_data/slice3/ML_data/results_evaluator_"+str(percent)+".csv"
cmu_slice4 = "colmap_data/CMU_data/slice4/ML_data/results_evaluator_"+str(percent)+".csv"
cmu_slice6 = "colmap_data/CMU_data/slice6/ML_data/results_evaluator_"+str(percent)+".csv"
cmu_slice10 = "colmap_data/CMU_data/slice10/ML_data/results_evaluator_"+str(percent)+".csv"
cmu_slice11 = "colmap_data/CMU_data/slice11/ML_data/results_evaluator_"+str(percent)+".csv"
coop_slice1 = "colmap_data/Coop_data/slice1/ML_data/results_evaluator_"+str(percent)+".csv"

cmu_slice3_PM = "colmap_data/CMU_data/slice3/ML_data/results_evaluator_"+str(percent)+"_predicting_matchability.csv"
cmu_slice4_PM = "colmap_data/CMU_data/slice4/ML_data/results_evaluator_"+str(percent)+"_predicting_matchability.csv"
cmu_slice6_PM = "colmap_data/CMU_data/slice6/ML_data/results_evaluator_"+str(percent)+"_predicting_matchability.csv"
cmu_slice10_PM = "colmap_data/CMU_data/slice10/ML_data/results_evaluator_"+str(percent)+"_predicting_matchability.csv"
cmu_slice11_PM = "colmap_data/CMU_data/slice11/ML_data/results_evaluator_"+str(percent)+"_predicting_matchability.csv"
coop_slice1_PM = "colmap_data/Coop_data/slice1/ML_data/results_evaluator_"+str(percent)+"_predicting_matchability.csv"

results_NN  = [cmu_slice3, cmu_slice4, cmu_slice6, cmu_slice10, cmu_slice11, coop_slice1]
results_PM  = [cmu_slice3_PM, cmu_slice4_PM, cmu_slice6_PM, cmu_slice10_PM, cmu_slice11_PM, coop_slice1_PM]
dataset_names = ["CMU Slice 3", "CMU Slice 4", "CMU Slice 6", "CMU Slice 10", "CMU Slice 11", "Coop Slice 1"]

header = ['Model', 'Inliers (%)', 'Outliers (%)', 'Cons. Iterations',
          'Cons. Time (s)', 'Feature M. Time (s)', 'Total Time (s)',
          'Trans. Error (m)', 'Rotation Error (o)']
csv_newline = ['']
model_names = list(names_dict.keys())

with open(csv_output, 'w', encoding='UTF8', newline='') as f:
    print("Writing to: " + csv_output)
    writer = csv.writer(f)
    for i in range(len(dataset_names)):
        model_name_idx = 0
        dataset_name = [dataset_names[i]] + [''] * 8 #just creating a row
        writer.writerow(dataset_name)
        # i is the number of slice
        result_NN_csv_file = results_NN[i] #this includes random and baseline too (should be in prepared data too)
        result_PM_csv_file = results_PM[i]

        writer.writerow(header)
        # NN results
        with open(result_NN_csv_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader: #loop through all NN models
                model_name = model_names[model_name_idx]
                inliers = float(row[0])
                outliers = float(row[1])
                total = inliers + outliers
                inliers_percentage = int(np.around(inliers * 100 / total))
                outliers_percentage = int(np.around(outliers * 100 / total))
                assert(inliers_percentage + outliers_percentage == 100)
                # formating them a bit better
                row[0] = str(inliers_percentage)
                row[1] = str(outliers_percentage)
                row[2] = str(np.around(float(row[2])))
                row[3] = ('%.3f' % float(row[3]))
                row[4] = ('%.3f' % float(row[4]))
                row[5] = ('%.3f' % float(row[5]))
                row[6] = ('%.2f' % float(row[6]))
                row[7] = ('%.2f' % float(row[7]))
                row.insert(0, model_name)
                writer.writerow(row)
                model_name_idx += 1

        # PM results
        with open(result_PM_csv_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            model_name = model_names[model_name_idx]
            row = next(csv_reader) #only pick the first row - no need for the rest (random and baseline, we already added them)
            inliers = float(row[0])
            outliers = float(row[1])
            total = inliers + outliers
            inliers_percentage = int(np.around(inliers * 100 / total))
            outliers_percentage = int(np.around(outliers * 100 / total))
            assert (inliers_percentage + outliers_percentage == 100)
            # formating them a bit better
            row[0] = str(inliers_percentage)
            row[1] = str(outliers_percentage)
            row[2] = str(np.around(float(row[2])))
            row[3] = ('%.3f' % float(row[3]))
            row[4] = ('%.3f' % float(row[4]))
            row[5] = ('%.3f' % float(row[5]))
            row[6] = ('%.2f' % float(row[6]))
            row[7] = ('%.2f' % float(row[7]))
            row.insert(0, model_name)
            writer.writerow(row)
            model_name_idx += 1 # increase for Match No Match

        writer.writerow(csv_newline)