# 05/07/2021 - re-used file for plots for the ML part of the project, run as: python3 plots/plots.py results.csv
# This script can run on my local laptop as it only needs a .csv file exported by excel.
# The data in the excel is first copy pasted from running: python3 print_eval_NN_results.py 5/10/..
# The data is imported in excel, from print_eval_NN_results.py, and each worksheet exported as .csv.
# The .csv is transferred to the server, scp /Users/alex/Desktop/results_20.csv ar2056@weatherwax.cs.bath.ac.uk:/homes/ar2056/fullpipeline/, and then used here
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

percentage = str(sys.argv[1])
results_path_csv = "results_" + str(percentage) + ".csv"
res = pd.read_csv(results_path_csv)
data = pd.DataFrame(res)

# 0 here starts from "CMU_slice3" not the headers
cmu_slice3 = data.iloc[0:23, 0:]
cmu_slice4 = data.iloc[24:47, 0:]
cmu_slice6 = data.iloc[48:71, 0:]
cmu_slice10 = data.iloc[72:95, 0:]
cmu_slice11 = data.iloc[96:119, 0:]
coop_slice1 = data.iloc[120:143, 0:]

all_dfs = [cmu_slice3, cmu_slice4, cmu_slice6, cmu_slice10, cmu_slice11, coop_slice1]

for df_slice in all_dfs:
    slice_name = df_slice.iloc[0,0]
    y = np.array(df_slice['t_err'].dropna())
    x = np.array(df_slice['rot_err'].dropna())
    ftm = np.array(df_slice['fm_time'].dropna())
    labels = np.array(df_slice.iloc[1:, 0])

    # this code is to remove random values from the graph
    # random_vals_index = -2 #use this to remove the random values as they mess up the scale, we already know random sucks no need to display it
    # y = np.delete(y, -2)
    # x = np.delete(x, -2)
    # ftm = np.delete(ftm, -2)
    # labels = np.delete(labels,-2)

    plt.figure(figsize=(12, 12), dpi=100)
    plt.suptitle('Error Per Method', fontsize=18)
    # plt.style.use('ggplot')
    plt.scatter(x, y, c=ftm, cmap='binary', edgecolors='k', s=170, alpha=0.7)
    plt.xlabel('Rotation Error (degrees)', fontsize=16)
    plt.ylabel('Translation Error (meters)', fontsize=16)
    plt.colorbar()

    # plt.xlim([0, 10]) # degrees - rotation
    # plt.ylim([0, 0.5]) # meters - translation
    # for i, txt in enumerate(labels):
    #     plt.annotate(txt, (x[i], y[i]))

    save_path = os.path.join("plots/", slice_name+"_"+percentage+".png")
    plt.savefig(save_path)
    plt.clf()
