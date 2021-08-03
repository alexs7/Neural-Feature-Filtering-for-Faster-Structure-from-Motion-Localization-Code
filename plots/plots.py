# 05/07/2021 - re-used file for plots for the ML part of the project, run as: python3 plots/plots.py
# This script can run on my local laptop as it only needs a .csv file exported by excel.
# The data in the excel is first copy pasted from running: python3 print_eval_NN_results.py 5/10/..
# The data is imported in excel, from print_eval_NN_results.py, and each worksheet exported as .csv.

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

plt.style.use('ggplot')
print(plt.style.available)

plt.figure(figsize=(12, 12), dpi=100)

names_dict = {
"Class, top mtchs" : "Classifier % matches",
"Class, all mtchs" : "Classifier using all matches",
"C & R, s.p.i" : "Classifier and Regressor w/ score per image",
"C & R, s.p.s" : "Classifier and Regressor w/ score per session",
"C & R, s.p.v" : "Classifier and Regressor w/ visibility score",
"R, s.p.i" : "Regressor w/ score per image",
"R, s.p.s" : "Regressor w/ score per session",
"R, s.p.v" : "Regressor w/ visibility score",
"CB, s.p.i" : "Combined w/ score per image",
"CB, s.p.s" : "Combined w/ score per session",
"CB, s.p.v" : "Combined w/ visibility score",
"Rd C & R s.p.i" : "Classifier and Regressor w/ score per image, modified RANSAC",
"Rd C & R s.p.s" : "Classifier and Regressor w/ score per session, modified RANSAC",
"Rd C & R s.p.v" : "Classifier and Regressor w/ visibility score, modified RANSAC",
"PRSC R, s.p.i" : "Regressor w/ score per image, PROSAC",
"PRSC R, s.p.s" : "Regressor w/ score per session, PROSAC",
"PRSC R, s.p.v" :  "Regressor w/ visibility score, PROSAC",
"PRSC CB, s.p.i" : "Combined w/ score per image, PROSAC",
"PRSC CB, s.p.s" : "Combined w/ score per session, PROSAC",
"PRSC CB, s.p.v" : "Combined w/ visibility score, PROSAC",
"Rndm 10%" : "Random feature case",
"All (~800)" : "Baseline using all features"
}

percentages = [5,10,15,20,50]
for percentage in percentages:
    results_path_csv = "plots/results_" + str(percentage) + ".csv"
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
        random_vals_index = -2 #use this to remove the random values as they mess up the scale, we already know random sucks no need to display it
        y = np.delete(y, -2)
        x = np.delete(x, -2)
        ftm = np.delete(ftm, -2)
        ftm = ftm * 1000
        labels = np.delete(labels,-2)

        plt.title('Error Per Method', fontsize=18)
        plt.scatter(x, y, c=ftm, cmap='binary', edgecolors='k', s=100, alpha=0.7)
        plt.xlabel('Rotation Error (degrees)', fontsize=16)
        plt.ylabel('Translation Error (meters)', fontsize=16)
        plt.colorbar()

        # limits
        plt.xlim([0, 3]) # degrees - rotation
        plt.ylim([0, 0.5]) # meters - translation
        for i, txt in enumerate(labels):
            label_key = txt #just for readability
            plt.annotate(names_dict[label_key], (x[i] + 0.01, y[i] + 0.007), fontsize=14)

        save_path = os.path.join("plots/", slice_name+"_"+str(percentage)+".png")
        plt.savefig(save_path)
        plt.clf()
