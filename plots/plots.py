# 05/07/2021 - re-used file for plots for the ML part of the project, run as: python3 plots/plots.py
# This script can run on my local laptop as it only needs a .csv file exported by excel.
# The data in the excel is first copy pasted from running: python3 print_eval_NN_results.py 5/10/..
# The data is imported in excel, from print_eval_NN_results.py, and each worksheet exported as .csv.

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
import numpy as np
import random

fig = plt.figure()

plt.style.use('ggplot')
print(plt.style.available)

# plt.figure(figsize=(8, 8), dpi=100)

names_dict = {
"Class, top mtchs" : "Classifier w/ top 10% matches",
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
"Rd C & R s.p.i" : "Class. and Regr. w/ score per image, dist. RANSAC",
"Rd C & R s.p.s" : "Class. and Regr. w/ score per session, dist. RANSAC",
"Rd C & R s.p.v" : "Class. and Regr. w/ visibility score, dist. RANSAC",
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

        # This code has to be tweak for each slice. I did for CMU slice 4 and Coop. Same goes for limits, and dots label position.
        # to get the idx_to_use values comment out "plt.annotate(names_dict[labels[i]] + " idx: " + str(i), (x[i] + 0.01, y[i] + 0.007), fontsize=8)"
        # idx_to_use_cmu_slice_4 = [8, 10, 9, 20, 0, 7]
        idx_to_use_coop = [20,12,7,11,13,8]
        x = np.array(x)[idx_to_use_coop]
        y = np.array(y)[idx_to_use_coop]
        labels = np.array(labels)[idx_to_use_coop]
        ftm = np.array(ftm)[idx_to_use_coop]
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red', 'tab:cyan', 'tab:pink', 'tab:olive']

        plt.title('Error Per Method')

        for i in range(len(labels)): # x,y,labels have the same order
            plt.scatter(x[i], y[i], s=100, alpha=0.75, c=colors[i], label=names_dict[labels[i]])

        plt.xlabel('Rotation Error (degrees)')
        plt.ylabel('Translation Error (meters)')
        # plt.colorbar()
        plt.legend(loc='best', framealpha=1, shadow=True)

        if(slice_name == 'Coop Data'): # need to use different limits
            # limits
            print('Coop data..')
            # plt.xlim([2, 3.5]) # degrees - rotation
            # plt.ylim([0.015, 0.035]) # meters - translation
        else:
            plt.xlim([0, 3]) # degrees - rotation
            plt.ylim([0, 0.5]) # meters - translation

        for i, txt in enumerate(labels):
            label_key = txt #just for readability
            # cmu slice 4
            # if(label_key == 'CB, s.p.v'): #manually positing them
            #     plt.annotate(str(round(ftm[i], 1)) + "ms", (x[i] - 0.09, y[i] + 0.01), fontsize=8) #for ftm
            # else:
            #     plt.annotate(str(round(ftm[i], 1)) + "ms", (x[i] + 0.01, y[i] + 0.01), fontsize=8)  # for ftm
            # coop
            if(label_key == "Rd C & R s.p.i"):
                plt.annotate(str(round(ftm[i], 1)) + "ms", (x[i] - 0.05, y[i] + 0.0005), fontsize=9)  # for ftm
                continue
            if (label_key == "CB, s.p.i"):
                plt.annotate(str(round(ftm[i], 1)) + "ms", (x[i] - 0.06, y[i] + 0.0003), fontsize=9)  # for ftm
                continue

            plt.annotate(str(round(ftm[i], 1)) + "ms", (x[i] + 0.01, y[i] + 0.0004), fontsize=9)  # for ftm
            # plt.annotate(" idx: " + str(i), (x[i], y[i]), fontsize=8) #for names debug

        # plt.tight_layout()
        fig.set_tight_layout(True)
        save_path = os.path.join("plots/", slice_name+"_"+str(percentage)+".pdf")
        plt.savefig(save_path)
        plt.clf()

# exit()
# # continue from here
# # df here stands for "dataframe"
# # Plot all best performing methods per dataset and the baseline (for the 10% case)
# plt.cla()
# results_path_csv = "plots/results_10.csv"
# res = pd.read_csv(results_path_csv)
# data = pd.DataFrame(res)
# 
# # 0 here starts from "CMU_slice3" not the headers
# cmu_slice3 = data.iloc[0:23, 0:]
# cmu_slice4 = data.iloc[24:47, 0:]
# cmu_slice6 = data.iloc[48:71, 0:]
# cmu_slice10 = data.iloc[72:95, 0:]
# cmu_slice11 = data.iloc[96:119, 0:]
# coop_slice1 = data.iloc[120:143, 0:]
# 
# all_dfs = [cmu_slice3, cmu_slice4, cmu_slice6, cmu_slice10, cmu_slice11, coop_slice1]
# 
# for df_slice in all_dfs:
#     slice_name = df_slice.iloc[0,0]
# 
#     y = np.array(df_slice['t_err'].dropna())
#     x = np.array(df_slice['rot_err'].dropna())
#     ftm = np.array(df_slice['fm_time'].dropna())
#     labels = np.array(df_slice.iloc[1:, 0])
# 
#     # this code is to remove random values from the graph
#     random_vals_index = -2 #use this to remove the random values as they mess up the scale, we already know random sucks no need to display it
#     y = np.delete(y, -2)
#     x = np.delete(x, -2)
#     ftm = np.delete(ftm, -2)
#     ftm = ftm * 1000
#     labels = np.delete(labels,-2)
# 
#     plt.title('Error Per Method')
#     plt.scatter(x, y, c=ftm, cmap='binary', edgecolors='k', s=100, alpha=0.7)
#     plt.xlabel('Rotation Error (degrees)')
#     plt.ylabel('Translation Error (meters)')
#     plt.colorbar()
# 
#     # limits
#     plt.xlim([0, 3]) # degrees - rotation
#     plt.ylim([0, 0.5]) # meters - translation
#     for i, txt in enumerate(labels):
#         label_key = txt #just for readability
#         plt.annotate(names_dict[label_key], (x[i] + 0.01, y[i] + 0.007), fontsize=14)
# 
#     save_path = os.path.join("plots/", slice_name+"_"+str(percentage)+".png")
#     plt.savefig(save_path)
#     plt.clf()
