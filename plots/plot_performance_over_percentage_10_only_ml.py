# Added for the ml part.
# The values here are copy pated by the excel file.
# this file was copied from plot_performance_over_percentages_ml.py to work only on the 10% case
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

plt.style.use('ggplot')
print(plt.style.available)

# These are exported files from the excel spreadsheet
results_10_path_csv = "plots/results_10.csv"

res = pd.read_csv(results_10_path_csv)
data_10 = pd.DataFrame(res)

# 0 here starts from "CMU_slice3" not the headers
cmu_slice3_10 = data_10.iloc[0:23, 0:]
cmu_slice4_10 = data_10.iloc[24:47, 0:]
cmu_slice6_10 = data_10.iloc[48:71, 0:]
cmu_slice10_10 = data_10.iloc[72:95, 0:]
cmu_slice11_10 = data_10.iloc[96:119, 0:]
coop_slice1_10 = data_10.iloc[120:143, 0:]

# datasets with percentages in them
all_cmu_slice3 = [cmu_slice3_10]
all_cmu_slice4 = [cmu_slice4_10]
all_cmu_slice6 = [cmu_slice6_10]
all_cmu_slice10 = [cmu_slice10_10]
all_cmu_slice11 = [cmu_slice11_10]
all_coop_slice1 = [coop_slice1_10]

#all_datasets
datasets = [all_cmu_slice3, all_cmu_slice4, all_cmu_slice6, all_cmu_slice10, all_cmu_slice11, all_coop_slice1]

# use for later, graphs
cmu_only_datasets = [all_cmu_slice3, all_cmu_slice4, all_cmu_slice6, all_cmu_slice10, all_cmu_slice11]
coop_only_dataset = [all_coop_slice1]

print("Applying Z-Transform")

all_slices_str = ["all_cmu_slice3", "all_cmu_slice4", "all_cmu_slice6", "all_cmu_slice10", "all_cmu_slice11", "all_coop_slice1"]
all_slices_str_readable = ["CMU Slice3", "CMU Slice4", "CMU Slice6", "CMU Slice10", "CMU Slice11", "Retail Shop"]
percentages_str = ["10%"]

# comment out which on you want to use
model_indices = np.delete(np.arange(1,23), [1,20, 21]) #remove the classifier trained on all (1), baseline (21) and random (20) - for paper
# model_indices = np.arange(1,23) #all models

best_methods = {}
all_slices_str_idx = 0
for dataset in datasets:
    dataset_name = all_slices_str[all_slices_str_idx]
    print("Dataset: " + dataset_name)
    percentages_str_idx = 0
    percentages_results_arr = np.array([all_slices_str[all_slices_str_idx]])
    percentages_results_arr_csv = np.array([all_slices_str[all_slices_str_idx]]) #duplicate for csv saving
    percentage = dataset[0] #get the 10% case

    # Choose here what to optimise for!
    # For example if you want to optimise for translation and rotation, comment out the other ones
    # If you optimise for translation and rotation beware that you have to modify "model_indices" too to remove the baseline, otherwise the baseline always wins

    conc_times_all_ml_methods = percentage.iloc[model_indices, 4]
    fm_times_all_ml_methods = percentage.iloc[model_indices, 5]
    t_err_all_ml_methods = percentage.iloc[model_indices, 7]
    rot_err_all_ml_methods = percentage.iloc[model_indices, 8]

    # conc_times_all_ml_methods_z_transformed = (conc_times_all_ml_methods - conc_times_all_ml_methods.mean() ) / conc_times_all_ml_methods.std()
    # fm_times_all_ml_methods_z_transformed = (fm_times_all_ml_methods - fm_times_all_ml_methods.mean() ) / fm_times_all_ml_methods.std()
    t_err_all_ml_methods_z_transformed = (t_err_all_ml_methods - t_err_all_ml_methods.mean() ) / t_err_all_ml_methods.std()
    rot_err_ml_methods_z_transformed = (rot_err_all_ml_methods - rot_err_all_ml_methods.mean() ) / rot_err_all_ml_methods.std()

    # pd.concat([t_err_all_ml_methods, rot_err_all_ml_methods], axis=1)
    all_metrics_ml_methods_z_transformed = pd.concat([#conc_times_all_ml_methods_z_transformed,
                                                      #fm_times_all_ml_methods_z_transformed,
                                                      t_err_all_ml_methods_z_transformed,
                                                      rot_err_ml_methods_z_transformed], axis=1)

    min_idx = all_metrics_ml_methods_z_transformed.mean(axis=1).rank(ascending=True).idxmin()
    best_method = percentage['method'][min_idx]

    print(" percentage: " + percentages_str[percentages_str_idx])
    print("  best method: " + best_method)

    best_methods[dataset_name] = best_method

    all_slices_str_idx += 1
    print()

# np.save("plots/best_methods_percentage_10.npy", best_methods)

print()
print("Saving Graphs..")

print("Bar Charts CMU..")
dataset_idx = 0
fm_times_best_model = np.array([])
t_err_best_model = np.array([])
rot_err_best_model = np.array([])
fm_times_baseline = np.array([])
t_err_baseline = np.array([])
rot_err_baseline = np.array([])
baseline_method = 'All (~800)'

for dataset_name, dataset_best_method in best_methods.items():
    print(" Bar Chart for " + dataset_name + " best model only")
    print("  Best Method is: " + dataset_best_method)
    print("dataset_idx " + str(dataset_idx))
    dataset = datasets[dataset_idx]

    # these will be the same for each percentage - because baseline, and classifier all (best model) always uses all the features
    dataset_temp = dataset[0] #doesnt matter which one here (0 is the 5% case)

    fm_times_baseline = np.append(fm_times_baseline, dataset_temp.loc[dataset_temp['method'] == baseline_method]['fm_time'] * 1000)  # convert to milliseconds
    t_err_baseline = np.append(t_err_baseline, dataset_temp.loc[dataset_temp['method'] == baseline_method]['t_err'])
    rot_err_baseline = np.append(rot_err_baseline, dataset_temp.loc[dataset_temp['method'] == baseline_method]['rot_err'])

    fm_times_best_model = np.append(fm_times_best_model, dataset_temp.loc[dataset_temp['method'] == dataset_best_method]['fm_time'] * 1000)  # convert to milliseconds
    t_err_best_model = np.append(t_err_best_model, dataset_temp.loc[dataset_temp['method'] == dataset_best_method]['t_err'])
    rot_err_best_model = np.append(rot_err_best_model, dataset_temp.loc[dataset_temp['method'] == dataset_best_method]['rot_err'])

    dataset_idx += 1

#ft matching times for all CMU slices in one bar chart
plt.figure(0)
ind = np.arange(0,3*len(fm_times_baseline[0:5]), 3)
width = 0.9
plt.title("Feature Matching Value")
plt.bar(ind, fm_times_baseline[0:5], width, label='Baseline Case per dataset')
plt.bar(ind + width, fm_times_best_model[0:5], width, label='Best Model per dataset')
plt.ylabel('Time (ms)', fontsize=10)
plt.xticks(ind + width/2, (all_slices_str_readable[0:-1]))
plt.legend(loc='best', framealpha=1, fontsize=10, shadow = True)
plt.savefig("plots/fm_times_all_cmu_bar_percentage_percentage_10.pdf")
plt.cla()

#t_err times for all CMU slices in one bar chart
plt.figure(1)
width = 0.9
plt.title("Translation Error")
plt.bar(ind, t_err_baseline[0:5], width, label='Baseline Case per dataset')
plt.bar(ind + width, t_err_best_model[0:5], width, label='Best Model per dataset')
plt.ylabel('Translation Error (m)', fontsize=10)
plt.xticks(ind + width/2, (all_slices_str_readable[0:-1]))
plt.legend(loc='best', framealpha=1, fontsize=10, shadow = True)
plt.savefig("plots/t_err_all_cmu_bar_percentage_percentage_10.pdf")
plt.cla()

#rot_err times for all CMU slices in one bar chart
plt.figure(2)
width = 0.9
plt.title("Rotation Error")
plt.bar(ind, rot_err_baseline[0:5], width, label='Baseline Case per dataset')
plt.bar(ind + width, rot_err_best_model[0:5], width, label='Best Model per dataset')
plt.ylabel('Rotation Error (degrees)', fontsize=10)
plt.xticks(ind + width/2, (all_slices_str_readable[0:-1]))
plt.legend(loc='best', framealpha=1, fontsize=10, shadow = True)
plt.savefig("plots/rot_err_all_cmu_bar_percentage_percentage_10.pdf")
plt.cla()

print("Bar Charts Coop.. - printing these for table not using bar charts")

print("Retail Shop Error Baseline VS Best Model:")

print("Baseline (Feature Matching Time/Translation/Rotation): " + str(fm_times_baseline[5]) + " / " + str(t_err_baseline[5]) + " / " + str(rot_err_baseline[5]))
print("Best Model (Feature Matching Time/Translation/Rotation/): " + str(fm_times_best_model[5]) + " / " + str(t_err_best_model[5]) + " / " + str(rot_err_best_model[5]))


print("Dots graphs..")
print("Run plots.py for dots.. troOolling!")
