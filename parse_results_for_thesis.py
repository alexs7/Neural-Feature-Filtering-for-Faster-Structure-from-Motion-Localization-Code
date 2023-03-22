#  Run this file after the learned_models_pose_data.py file to parse its .csv file (run it wherever the .csv file is for now it is on CYENS)
#  Also this file is used to report various metadata such as the imbalanced ratio of the PM data.
#  This file is supposed to be edited frequently
#  Will also prin latex tables for the thesis
#  some manual work is needed to get the latex tables to look good
# This file has breakpoints all over the place as you used them to examine the variable values print them etc

import os
import sys
import numpy as np
import pandas as pd
from database import COLMAPDatabase
from parameters import Parameters

def print_methods_metric_for_dataset(mnm_vals, nf_vals, pm_vals, label, caption, title):
    print()
    print("\\begin{table}[h]")
    print("\\caption{\\label{tab:"+label+"}%")
    print(caption)
    print("}\\vspace{0.5em}")
    print("\\centerline{")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")

    print("Dataset & \\multicolumn{3}{c}{"+title+"} \\\\ \\midrule")
    print("Method & MnM & NF & PM  \\\\ \\midrule")

    # These are average values for each method for each slice for CMU, for LaMAR, and for RetailShop (not average for RetailShop as there is only one slice)
    # You bold the values in the thesis code
    mnm_vals.iloc[1:] = mnm_vals.iloc[1:].astype(np.float64)
    nf_vals.iloc[1:] = nf_vals.iloc[1:].astype(np.float64)
    pm_vals.iloc[1:] = pm_vals.iloc[1:].astype(np.float64)
    print(f"Translation Error[m] & {mnm_vals[1]:.2f} & {nf_vals[1]:.2f} & {pm_vals[1]:.2f} \\\\ ")
    print(f"Rotation Error[°] & {mnm_vals[2]:.2f} & {nf_vals[2]:.2f} & {pm_vals[2]:.2f} \\\\ ")
    print(f"Features Reduction[\\%] & {mnm_vals[3]:.2f} & {nf_vals[3]:.2f} & {pm_vals[3]:.2f} \\\\")
    print(f"Feature Matching Time (ms) & {'{:,}'.format(int(mnm_vals[4] * 1000))} & {'{:,}'.format(int(nf_vals[4] * 1000))} & {'{:,}'.format(int(pm_vals[4] * 1000))} \\\\ ")
    print(f"Consensus Time (ms) & {mnm_vals[5] * 1000:.0f} & {nf_vals[5] * 1000:.0f} & {pm_vals[5] * 1000:.2f} \\\\ ")
    print(f"mAA[\\%] & {mnm_vals[6] * 100:.2f} & {nf_vals[6] * 100:.2f} & {pm_vals[6] * 100:.2f} \\\\ ")
    print(f"Degenerate Poses No. & {int(mnm_vals[7])} & {int(nf_vals[7])} & {int(pm_vals[7])} \\\\ ")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")  # for print("\\centerline{")
    print("\\end{table}")

def print_methods_metric_for_dataset_binary(mnm_vals, nf_vals, pm_vals, label, caption, title):
    print()
    print("\\begin{table}[h]")
    print("\\caption{\\label{tab:"+label+"}%")
    print(caption)
    print("}\\vspace{0.5em}")
    print("\\centerline{")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")

    print("Dataset & \\multicolumn{3}{c}{"+title+"} \\\\ \\midrule")
    print("Method & MnM & NF & PM  \\\\ \\midrule")

    # These are average values for each method for each slice for CMU, for LaMAR, and for RetailShop (not average for RetailShop as there is only one slice)
    # You bold the values in the thesis code
    mnm_vals.iloc[1:] = mnm_vals.iloc[1:].astype(np.float64)
    nf_vals.iloc[1:] = nf_vals.iloc[1:].astype(np.float64)
    pm_vals.iloc[1:] = pm_vals.iloc[1:].astype(np.float64)
    print(f"Precision (Positive Class) & {mnm_vals[1]:.2f} & {nf_vals[1]:.2f} & {pm_vals[1]:.2f} \\\\ ")
    print(f"Recall (Positive Class) & {mnm_vals[2]:.2f} & {nf_vals[2]:.2f} & {pm_vals[2]:.2f} \\\\ ")
    print(f"F1 Score & {mnm_vals[3]:.2f} & {nf_vals[3]:.2f} & {pm_vals[3]:.2f} \\\\ ")
    print(f"True Negatives [\\%] & {mnm_vals[4]:.2f} & {nf_vals[4]:.2f} & {pm_vals[4]:.2f} \\\\ ")
    print(f"False Negatives [\\%] & {mnm_vals[5]:.2f} & {nf_vals[5]:.2f} & {pm_vals[5]:.2f} \\\\ ")
    print(f"False Positives [\\%] & {mnm_vals[6]:.2f} & {nf_vals[6]:.2f} & {pm_vals[6]:.2f} \\\\ ")
    print(f"True Positives [\\%] & {mnm_vals[7]:.2f} & {nf_vals[7]:.2f} & {pm_vals[7]:.2f} \\\\ ")
    print(f"Balanced Accuracy [\\%] & {mnm_vals[8]:.2f} & {nf_vals[8]:.2f} & {pm_vals[8]:.2f} \\\\ ")
    print(f"Accuracy [\\%] & {mnm_vals[9]:.2f} & {nf_vals[9]:.2f} & {pm_vals[9]:.2f} \\\\ ")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")  # for print("\\centerline{")
    print("\\end{table}")

root_path = "/media/iNicosiaData/engd_data/"

# Pose data (trans error, rot error, features reduction, feature matching time, consensus time, mAA, degenerate cases)
file_pose_data = sys.argv[1]
# TP, FP, FN, TN, Precision, Recall, F1, Accuracy etc
file_binary_data = sys.argv[2]

print("Reading .csv")
# Also the name of the last column is "Degenerate Cases" but it is missing - can add it manually
results_df = pd.read_csv(os.path.join(root_path, file_pose_data), header=None)

mnm_df_cmu = pd.DataFrame()
nf_df_cmu = pd.DataFrame()
pm_df_cmu = pd.DataFrame()
mnm_df_retail = pd.DataFrame()
nf_df_retail = pd.DataFrame()
pm_df_retail = pd.DataFrame()
mnm_df_lamar = pd.DataFrame()
nf_df_lamar = pd.DataFrame()
pm_df_lamar = pd.DataFrame()

# Collect all data per method and dataset
for i in range(len(results_df)):
    row = results_df.iloc[i]
    if (pd.isnull(row[0])):
        continue # skip empty rows
    if("slice" in row[0]): #i changes here
        # MnM datat
        mnm_data = results_df.iloc[i+1]
        mnm_df_cmu = pd.concat([mnm_df_cmu, mnm_data], axis=1)
        nf_data = results_df.iloc[i + 2]
        nf_df_cmu = pd.concat([nf_df_cmu, nf_data], axis=1)
        pm_data = results_df.iloc[i + 4] #skipping nf small
        pm_df_cmu = pd.concat([pm_df_cmu, pm_data], axis=1)
    if ("RetailShop" in row[0]): #i changes here
        mnm_data = results_df.iloc[i + 1]
        mnm_df_retail = pd.concat([mnm_df_retail, mnm_data], axis=1)
        nf_data = results_df.iloc[i + 2]
        nf_df_retail = pd.concat([nf_df_retail, nf_data], axis=1)
        pm_data = results_df.iloc[i + 4]  # skipping nf small
        pm_df_retail = pd.concat([pm_df_retail, pm_data], axis=1)
    if ("LIN" in row[0] or "CAB" in row[0] or "HGE" in row[0]): #i changes here
        mnm_data = results_df.iloc[i + 1]
        mnm_df_lamar = pd.concat([mnm_df_lamar, mnm_data], axis=1)
        nf_data = results_df.iloc[i + 2]
        nf_df_lamar = pd.concat([nf_df_lamar, nf_data], axis=1)
        pm_data = results_df.iloc[i + 4]  # skipping nf small
        pm_df_lamar = pd.concat([pm_df_lamar, pm_data], axis=1)

print("Averages for CMU - MnM")
print(mnm_df_cmu.iloc[1:].astype(np.float64).mean(axis=1))

print("Averages for CMU - NF")
print(nf_df_cmu.iloc[1:].astype(np.float64).mean(axis=1))

print("Averages for CMU - PM")
print(pm_df_cmu.iloc[1:].astype(np.float64).mean(axis=1))

print("Values for RetailShop - MnM")
print(mnm_df_retail.iloc[1:].astype(np.float64).mean(axis=1))

print("Values for RetailShop - NF")
print(nf_df_retail.iloc[1:].astype(np.float64).mean(axis=1))

print("Values for RetailShop - PM")
print(pm_df_retail.iloc[1:].astype(np.float64).mean(axis=1))

print("Averages for Lamar - MnM")
print(mnm_df_lamar.iloc[1:].astype(np.float64).mean(axis=1))

print("Averages for Lamar - NF")
print(nf_df_lamar.iloc[1:].astype(np.float64).mean(axis=1))

print("Averages for Lamar - PM")
print(pm_df_lamar.iloc[1:].astype(np.float64).mean(axis=1))

print("Printing reduction % of features, the % that is removed")
# data for teaser figure chapter 5
# this is ugly but it works (same value as spreadsheet)
mean_nf_reduction = np.append(np.append(nf_df_cmu.iloc[1:].astype(np.float64).values[2], nf_df_retail.iloc[1:].astype(np.float64).mean(axis=1).values[2]), (nf_df_lamar.iloc[1:].astype(np.float64).mean(axis=1).values[2])).mean()
mean_feature_matching_time_nf = np.append(np.append(nf_df_cmu.iloc[1:].astype(np.float64).values[4], nf_df_retail.iloc[1:].astype(np.float64).mean(axis=1).values[4]), (nf_df_lamar.iloc[1:].astype(np.float64).mean(axis=1).values[4])).mean()
mean_feature_matching_time_mnm = np.append(np.append(mnm_df_cmu.iloc[1:].astype(np.float64).values[4], mnm_df_retail.iloc[1:].astype(np.float64).mean(axis=1).values[4]), (mnm_df_lamar.iloc[1:].astype(np.float64).mean(axis=1).values[4])).mean()
mean_feature_matching_time_pm = np.append(np.append(pm_df_cmu.iloc[1:].astype(np.float64).values[4], pm_df_retail.iloc[1:].astype(np.float64).mean(axis=1).values[4]), (pm_df_lamar.iloc[1:].astype(np.float64).mean(axis=1).values[4])).mean()

breakpoint()

print(f"NF percentage reduction: {mean_nf_reduction}")
print(f"Mean feature time NF: {mean_feature_matching_time_nf}")
print(f"Mean feature time MnM: {mean_feature_matching_time_mnm}")
print(f"Mean feature time PM (lower but less accurate): {mean_feature_matching_time_pm}")

# Generate latex tables here 24/03/2023
print("------------------------------------ The mean datasets table values ------------------------------------")
cmu_mnm_mean_table_vals = mnm_df_cmu.iloc[1:].astype(np.float64).mean(axis=1)
cmu_nf_mean_table_vals = nf_df_cmu.iloc[1:].astype(np.float64).mean(axis=1)
cmu_pm_mean_table_vals = pm_df_cmu.iloc[1:].astype(np.float64).mean(axis=1)
retail_shop_mnm_mean_table_vals = mnm_df_retail.iloc[1:].astype(np.float64).mean(axis=1)
retail_shop_nf_mean_table_vals = nf_df_retail.iloc[1:].astype(np.float64).mean(axis=1)
retail_shop_pm_mean_table_vals = pm_df_retail.iloc[1:].astype(np.float64).mean(axis=1)
lamar_mnm_mean_table_vals = mnm_df_lamar.iloc[1:].astype(np.float64).mean(axis=1)
lamar_nf_mean_table_vals = nf_df_lamar.iloc[1:].astype(np.float64).mean(axis=1)
lamar_pm_mean_table_vals = pm_df_lamar.iloc[1:].astype(np.float64).mean(axis=1)

print()
print("\\begin{table} % [h]")
print("\\caption{\\label{tab:replace}%")
print("caption replace")
print("}\\vspace{0.5em}")
print("\\centerline{")
print("\\begin{tabular}{lccccccccc}")
print("\\toprule")

print("Dataset & \\multicolumn{3}{c}{CMU} & \\multicolumn{3}{c}{LaMAR} & \\multicolumn{3}{c}{Retail Shop} \\\\ \\midrule")
print("Method & MnM & NF & PM & MnM & NF & PM & MnM & NF & PM \\\\ \\midrule")

# These are average values for each method for each slice for CMU, for LaMAR, and for RetailShop (not average for RetailShop as there is only one slice)
# You bold the values in the thesis code
print(f"Trans. Er.[m/cm] & {cmu_mnm_mean_table_vals[1]:.2f} & {cmu_nf_mean_table_vals[1]:.2f} & {cmu_pm_mean_table_vals[1]:.2f} & {lamar_mnm_mean_table_vals[1]:.2f} & {lamar_nf_mean_table_vals[1]:.2f} & {lamar_pm_mean_table_vals[1]:.2f} & {retail_shop_mnm_mean_table_vals[1]*100:.2f} & {retail_shop_nf_mean_table_vals[1]*100:.2f} & {retail_shop_pm_mean_table_vals[1]*100:.2f} \\\\")
print(f"Rot. Er.[°] & {cmu_mnm_mean_table_vals[2]:.2f} & {cmu_nf_mean_table_vals[2]:.2f} & {cmu_pm_mean_table_vals[2]:.2f} & {lamar_mnm_mean_table_vals[2]:.2f} & {lamar_nf_mean_table_vals[2]:.2f} & {lamar_pm_mean_table_vals[2]:.2f} & {retail_shop_mnm_mean_table_vals[2]:.2f} & {retail_shop_nf_mean_table_vals[2]:.2f} & {retail_shop_pm_mean_table_vals[2]:.2f} \\\\")
print(f"Features Red.[\\%] & {cmu_mnm_mean_table_vals[3]:.2f} & {cmu_nf_mean_table_vals[3]:.2f} & {cmu_pm_mean_table_vals[3]:.2f} & {lamar_mnm_mean_table_vals[3]:.2f} & {lamar_nf_mean_table_vals[3]:.2f} & {lamar_pm_mean_table_vals[3]:.2f} & {retail_shop_mnm_mean_table_vals[3]:.2f} & {retail_shop_nf_mean_table_vals[3]:.2f} & {retail_shop_pm_mean_table_vals[3]:.2f} \\\\")
print(f"F.M. Time (ms) & {cmu_mnm_mean_table_vals[4]*1000:.0f} & {cmu_nf_mean_table_vals[4]*1000:.0f} & {cmu_pm_mean_table_vals[4]*1000:.0f} & {lamar_mnm_mean_table_vals[4]*1000:.0f} & {lamar_nf_mean_table_vals[4]*1000:.0f} & {lamar_pm_mean_table_vals[4]*1000:.0f} & {retail_shop_mnm_mean_table_vals[4]*1000:.0f} & {retail_shop_nf_mean_table_vals[4]*1000:.0f} & {retail_shop_pm_mean_table_vals[4]*1000:.0f} \\\\")
print(f"Consensus Time (ms) & {cmu_mnm_mean_table_vals[5]*1000:.0f} & {cmu_nf_mean_table_vals[5]*1000:.0f} & {cmu_pm_mean_table_vals[5]*1000:.0f} & {lamar_mnm_mean_table_vals[5]*1000:.0f} & {lamar_nf_mean_table_vals[5]*1000:.0f} & {lamar_pm_mean_table_vals[5]*1000:.0f} & {retail_shop_mnm_mean_table_vals[5]*1000:.0f} & {retail_shop_nf_mean_table_vals[5]*1000:.0f} & {retail_shop_pm_mean_table_vals[5]*1000:.0f} \\\\")
print(f"mAA[\\%] & {cmu_mnm_mean_table_vals[6]*100:.2f} & {cmu_nf_mean_table_vals[6]*100:.2f} & {cmu_pm_mean_table_vals[6]*100:.2f} & {lamar_mnm_mean_table_vals[6]*100:.2f} & {lamar_nf_mean_table_vals[6]*100:.2f} & {lamar_pm_mean_table_vals[6]*100:.2f} & {retail_shop_mnm_mean_table_vals[6]*100:.2f} & {retail_shop_nf_mean_table_vals[6]*100:.2f} & {retail_shop_pm_mean_table_vals[6]*100:.2f} \\\\")
# print(f"Degen. & {cmu_mnm_mean_table_vals[7]:.2f} & {cmu_nf_mean_table_vals[7]:.2f} & {cmu_pm_mean_table_vals[7]:.2f} & {lamar_mnm_mean_table_vals[7]:.2f} & {lamar_nf_mean_table_vals[7]:.2f} & {lamar_pm_mean_table_vals[7]:.2f} & {retail_shop_mnm_mean_table_vals[7]:.2f} & {retail_shop_nf_mean_table_vals[7]:.2f} & {retail_shop_pm_mean_table_vals[7]:.2f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("}")  # for print("\\centerline{")
print("\\end{table}")
print()

print("!------------------------------------ The mean datasets table values ------------------------------------")

print()

print("------------------------------------ The individual datasets table values (Appendix B)------------------------------------")
# The order is MnM, NF, PM
# CMU
slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                    "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
no_slices = mnm_df_cmu.shape[1]
for i in range(no_slices):
    if i == 10: #because of latex
        print("\\clearpage")
    slice_name = slices_names[i]
    slice_mnm_data = mnm_df_cmu.iloc[:, i]
    slice_pm_data = pm_df_cmu.iloc[:, i]
    slice_nf_data = nf_df_cmu.iloc[:, i]
    caption = f"The table shows the error metrics for the CMU slice {i+2} dataset." # +2 because slice2 is the first slice
    print_methods_metric_for_dataset(slice_mnm_data, slice_nf_data, slice_pm_data, f"slice_{i}_model_errors", caption, title=f"CMU slice {i+2}")

print("\\clearpage")

# Lamar
lamar_names = ["LIN", "CAB", "HGE"] #same order as in the result file
no_sub_datasets = mnm_df_lamar.shape[1]

for i in range(no_sub_datasets):
    lamar_name = lamar_names[i]
    lamar_mnm_data = mnm_df_lamar.iloc[:, i]
    lamar_pm_data = pm_df_lamar.iloc[:, i]
    lamar_nf_data = nf_df_lamar.iloc[:, i]
    caption = f"The table shows the error metrics for the LaMAR {lamar_name} dataset."
    print_methods_metric_for_dataset(lamar_mnm_data, lamar_nf_data, lamar_pm_data, f"{lamar_name}_model_errors", caption, title=f"LaMAR {lamar_name}")

print("\\clearpage")

# Retail Shop
retail_shop_name = "Retail Shop"
retail_shop_mnm_data = mnm_df_retail.iloc[:, 0]
retail_shop_pm_data = pm_df_retail.iloc[:, 0]
retail_shop_nf_data = nf_df_retail.iloc[:, 0]
caption = f"The table shows the error metrics for the {retail_shop_name} dataset."
print_methods_metric_for_dataset(retail_shop_mnm_data, retail_shop_nf_data, retail_shop_pm_data, f"{retail_shop_name}_model_errors", caption, title=f"{retail_shop_name}")

print("!------------------------------------ The individual datasets table values (Appendix B)------------------------------------")

print()

print("------------------------------------ The individual datasets table BINARY values (Appendix B)------------------------------------")

print("Generating latex tables for binary data")

# TODO: Add code to generate Table for the NF bce model anf NF small model
# as of now you are manually writing latex.

print("Reading .csv")
# NOTE add a row above the results .csv file to show the dataset name
# Also the name of the last column is "Degenerate Cases" but it is missing - can add it manually
results_df = pd.read_csv(os.path.join(root_path, file_binary_data), header=None)

mnm_df_cmu_binary = pd.DataFrame()
nf_df_cmu_binary = pd.DataFrame()
pm_df_cmu_binary = pd.DataFrame()
mnm_df_retail_binary = pd.DataFrame()
nf_df_retail_binary = pd.DataFrame()
pm_df_retail_binary = pd.DataFrame()
mnm_df_lamar_binary = pd.DataFrame()
nf_df_lamar_binary = pd.DataFrame()
pm_df_lamar_binary = pd.DataFrame()

# Collect all data per method and dataset
for i in range(len(results_df)):
    row = results_df.iloc[i]
    if (pd.isnull(row[0])):
        continue # skip empty rows
    if("slice" in row[0]): #i changes here
        # MnM data
        mnm_data = results_df.iloc[i+1]
        mnm_df_cmu_binary = pd.concat([mnm_df_cmu_binary, mnm_data], axis=1)
        nf_data = results_df.iloc[i + 2]
        nf_df_cmu_binary = pd.concat([nf_df_cmu_binary, nf_data], axis=1)
        pm_data = results_df.iloc[i + 4] #skipping nf small
        pm_df_cmu_binary = pd.concat([pm_df_cmu_binary, pm_data], axis=1)
    if ("RetailShop" in row[0]): #i changes here
        mnm_data = results_df.iloc[i + 1]
        mnm_df_retail_binary = pd.concat([mnm_df_retail_binary, mnm_data], axis=1)
        nf_data = results_df.iloc[i + 2]
        nf_df_retail_binary = pd.concat([nf_df_retail_binary, nf_data], axis=1)
        pm_data = results_df.iloc[i + 4]  # skipping nf small
        pm_df_retail_binary = pd.concat([pm_df_retail_binary, pm_data], axis=1)
    if ("LIN" in row[0] or "CAB" in row[0] or "HGE" in row[0]): #i changes here
        mnm_data = results_df.iloc[i + 1]
        mnm_df_lamar_binary = pd.concat([mnm_df_lamar_binary, mnm_data], axis=1)
        nf_data = results_df.iloc[i + 2]
        nf_df_lamar_binary = pd.concat([nf_df_lamar_binary, nf_data], axis=1)
        pm_data = results_df.iloc[i + 4]  # skipping nf small
        pm_df_lamar_binary = pd.concat([pm_df_lamar_binary, pm_data], axis=1)

# The order is MnM, NF, PM
# CMU
slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                    "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
no_slices = mnm_df_cmu.shape[1] #same as binary
for i in range(no_slices):
    if i == 10: #because of latex
        print("\\clearpage")
    slice_name = slices_names[i]
    slice_mnm_data = mnm_df_cmu_binary.iloc[:, i]
    slice_pm_data = pm_df_cmu_binary.iloc[:, i]
    slice_nf_data = nf_df_cmu_binary.iloc[:, i]
    caption = f"The table shows the binary classifier metrics for the CMU slice {i+2} dataset." # +2 because slice2 is the first slice
    print_methods_metric_for_dataset_binary(slice_mnm_data, slice_nf_data, slice_pm_data, f"slice_{i}_binary_stats", caption, title=f"CMU slice {i+2}")

print("\\clearpage")

# Lamar
lamar_names = ["HGE", "CAB", "LIN"] #same order as in the binary result file
no_sub_datasets = mnm_df_lamar_binary.shape[1]

for i in range(no_sub_datasets):
    lamar_name = lamar_names[i]
    lamar_mnm_data = mnm_df_lamar_binary.iloc[:, i]
    lamar_pm_data = pm_df_lamar_binary.iloc[:, i]
    lamar_nf_data = nf_df_lamar_binary.iloc[:, i]
    caption = f"The table shows the binary classifier metrics for the LaMAR {lamar_name} dataset."
    print_methods_metric_for_dataset_binary(lamar_mnm_data, lamar_nf_data, lamar_pm_data, f"{lamar_name}_binary_stats", caption, title=f"LaMAR {lamar_name}")

print("\\clearpage")

# Retail Shop
retail_shop_name = "Retail Shop"
retail_shop_mnm_data = mnm_df_retail_binary.iloc[:, 0]
retail_shop_pm_data = pm_df_retail_binary.iloc[:, 0]
retail_shop_nf_data = nf_df_retail_binary.iloc[:, 0]
caption = f"The table shows the binary classifier metrics for the {retail_shop_name} dataset."
print_methods_metric_for_dataset_binary(retail_shop_mnm_data, retail_shop_nf_data, retail_shop_pm_data, f"{retail_shop_name}_binary_stats", caption, title=f"{retail_shop_name}")

print("!------------------------------------ The individual datasets table BINARY values (Appendix B)------------------------------------")

print("------------------------------------ The CMU table Feature Red. | FM. Time | Maa | values (For main text)------------------------------------")

no_slices = len(slices_names)
print()
print("\\begin{table}[h]")
print("\\caption{\\label{tab:cmu_reduction_fm_time_maa}%")
print("Replace")
print("}\\vspace{0.5em}")
print("\\centerline{")
print("\\begin{tabular}{lrrrrrrrrr}")
print("\\toprule")

print(" & \\multicolumn{3}{c}{Keypoint Reduction[\%]}  & \\multicolumn{3}{c}{F.M Time[ms]}  & \\multicolumn{3}{c}{MAA[\%]} \\\\ \\midrule")
print("Method & MnM & NF & PM & MnM & NF & PM & MnM & NF & PM \\\\ \\midrule")
for i in range(no_slices):
    slice_name = slices_names[i]
    slice_mnm_data = mnm_df_cmu.iloc[:, i]
    slice_pm_data = pm_df_cmu.iloc[:, i]
    slice_nf_data = nf_df_cmu.iloc[:, i]
    if(slice_mnm_data[6] <= slice_nf_data[6] and slice_pm_data[6] <= slice_nf_data[6]):
        print(f"{slice_name} & {slice_mnm_data[3]:.2f} & \\textbf{{{slice_nf_data[3]:.2f}}} & {slice_pm_data[3]:.2f} & {'{:,}'.format(int(slice_mnm_data[4]*1000))} & {'{:,}'.format(int(slice_nf_data[4]*1000))} & {'{:,}'.format(int(slice_pm_data[4]*1000))} & {slice_mnm_data[6]*100:.2f} & {slice_nf_data[6]*100:.2f} & {slice_pm_data[6]*100:.2f} \\\\")
    else:
        print(f"{slice_name} & {slice_mnm_data[3]:.2f} & {slice_nf_data[3]:.2f} & {slice_pm_data[3]:.2f} & {'{:,}'.format(int(slice_mnm_data[4]*1000))} & {'{:,}'.format(int(slice_nf_data[4]*1000))} & {'{:,}'.format(int(slice_pm_data[4]*1000))} & {slice_mnm_data[6]*100:.2f} & {slice_nf_data[6]*100:.2f} & {slice_pm_data[6]*100:.2f} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("}")  # for print("\\centerline{")
print("\\end{table}")
print()

print("!------------------------------------ The CMU table Feature Red. | FM. Time | Maa | values (For main text)------------------------------------")

print("Printing imbalanced data ratios..")

print("For PM's training data..")

print("LAMAR PM data..")
for dataset in ["HGE","CAB","LIN"]:
    base_path = os.path.join(root_path, "lamar", f"{dataset}_colmap_model")
    print("Base path: " + base_path)
    parameters = Parameters(base_path)
    file_identifier = f"{parameters.predicting_matchability_comparison_data_lamar_no_samples}_samples_opencv"
    training_data_db_path = os.path.join(parameters.predicting_matchability_comparison_data_full, f"training_data_{file_identifier}.db")
    training_data_db = COLMAPDatabase.connect(training_data_db_path)
    stats = training_data_db.execute("SELECT * FROM data").fetchall()
    matched = training_data_db.execute("SELECT * FROM data WHERE matched = 1").fetchall()
    unmatched = training_data_db.execute("SELECT * FROM data WHERE matched = 0").fetchall()

    print("Total descs: " + str(len(stats)))
    print("Total matched descs: " + str(len(matched)))
    print("Total unmatched descs: " + str(len(unmatched)))
    print("% of matched decs: " + str(len(matched) * 100 / len(stats)))

print("CMU PM data..")
cmu_mean_val = []
slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
for slice_name in slices_names:
    base_path = os.path.join(root_path, "cmu", f"{slice_name}", "exmaps_data")
    print("Base path: " + base_path)
    parameters = Parameters(base_path)
    training_data_db_path = os.path.join(parameters.predicting_matchability_comparison_data_full, f"training_data_1200_samples_opencv.db")
    training_data_db = COLMAPDatabase.connect(training_data_db_path)
    stats = training_data_db.execute("SELECT * FROM data").fetchall()
    matched = training_data_db.execute("SELECT * FROM data WHERE matched = 1").fetchall()
    unmatched = training_data_db.execute("SELECT * FROM data WHERE matched = 0").fetchall()

    print("Total descs: " + str(len(stats)))
    print("Total matched descs: " + str(len(matched)))
    print("Total unmatched descs: " + str(len(unmatched)))
    print("% of matched decs: " + str(len(matched) * 100 / len(stats)))
    cmu_mean_val.append(len(matched) * 100 / len(stats))

print("CMU mean % of matched decs:")
print(np.mean(cmu_mean_val))

print("Retail Shop PM data..")
base_path = os.path.join(root_path, "retail_shop", "slice1")
print("Base path: " + base_path)
parameters = Parameters(base_path)
training_data_db_path = os.path.join(parameters.predicting_matchability_comparison_data_full, f"training_data_1200_samples_opencv.db")
training_data_db = COLMAPDatabase.connect(training_data_db_path)
stats = training_data_db.execute("SELECT * FROM data").fetchall()
matched = training_data_db.execute("SELECT * FROM data WHERE matched = 1").fetchall()
unmatched = training_data_db.execute("SELECT * FROM data WHERE matched = 0").fetchall()

print("Total descs: " + str(len(stats)))
print("Total matched descs: " + str(len(matched)))
print("Total unmatched descs: " + str(len(unmatched)))
print("% of matched decs: " + str(len(matched) * 100 / len(stats)))

print("For my own 3D data..")

for dataset in ["HGE" , "CAB" , "LIN"]:
    base_path = os.path.join(root_path, "lamar", f"{dataset}_colmap_model")
    print("Base path: " + base_path)
    parameters = Parameters(base_path)
    percentages = np.loadtxt(os.path.join(parameters.base_path, "ML_data/", f"{dataset}_classes_percentage.txt"))

print("-> LaMAR")
arr_rounded = np.round(percentages, decimals=2)
print(arr_rounded)

mean_arr = np.empty([0,2])
slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
for slice_name in slices_names:
    base_path = os.path.join(root_path, "cmu", f"{slice_name}", "exmaps_data")
    parameters = Parameters(base_path)
    percentages = np.loadtxt(os.path.join(parameters.base_path, "ML_data/", f"CMU_{slice_name}_classes_percentage.txt"))
    mean_arr = np.append(mean_arr, np.array([percentages]), axis=0)

print("-> CMU")
arr_rounded = np.round(np.mean(mean_arr, axis=0), decimals=2)
print(arr_rounded)

base_path = os.path.join(root_path, "retail_shop", "slice1")
print("Base path: " + base_path)
parameters = Parameters(base_path)
dataset = "RetailShop"
percentages = np.loadtxt(os.path.join(parameters.base_path, "ML_data/", f"{dataset}_classes_percentage.txt"))

print("-> Retail Shop")
arr_rounded = np.round(percentages, decimals=2)
print(arr_rounded)

print("Done")