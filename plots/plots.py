# 05/07/2021 - re-used file for plots for the ML part of the project, run as: python3 plots/plots.py results.csv
# This script can run on my local laptop as it only needs a .csv file exported by excel.
# The data in the excel is first copy pasted from running: python3 print_eval_NN_results.py NN_results_excel.txt
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results_path_csv = sys.argv[1]
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
    labels = np.array(df_slice.iloc[1:, 0])

    random_vals_index = -2 #use this to remove the random values as they mess up the scale, we already know random sucks no need to display it
    y = np.delete(y,-2)
    x = np.delete(x,-2)
    labels = np.delete(labels,-2)

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    fig.suptitle('Error Per Method', fontsize=18)
    plt.xlabel('Rotation Error (degrees)', fontsize=16)
    plt.ylabel('Translation Error (meters)', fontsize=16)

    # plt.xlim([0, 10]) # degrees - rotation
    # plt.ylim([0, 0.5]) # meters - translation

    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]))

    save_path = os.path.join("plots/", slice_name+".png")
    plt.savefig(save_path)
