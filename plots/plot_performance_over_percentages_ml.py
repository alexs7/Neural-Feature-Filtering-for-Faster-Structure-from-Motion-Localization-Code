# Added for the ml part.
# The values here are copy pated by the excel file.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

plt.style.use('ggplot')
print(plt.style.available)

results_5_path_csv = "plots/results_5.csv"
results_10_path_csv = "plots/results_10.csv"
results_15_path_csv = "plots/results_15.csv"
results_20_path_csv = "plots/results_20.csv"
results_50_path_csv = "plots/results_50.csv"

res = pd.read_csv(results_5_path_csv)
data_5 = pd.DataFrame(res)
res = pd.read_csv(results_10_path_csv)
data_10 = pd.DataFrame(res)
res = pd.read_csv(results_15_path_csv)
data_15 = pd.DataFrame(res)
res = pd.read_csv(results_20_path_csv)
data_20 = pd.DataFrame(res)
res = pd.read_csv(results_50_path_csv)
data_50 = pd.DataFrame(res)

# 0 here starts from "CMU_slice3" not the headers
cmu_slice3_5 = data_5.iloc[0:23, 0:]
cmu_slice4_5 = data_5.iloc[24:47, 0:]
cmu_slice6_5 = data_5.iloc[48:71, 0:]
cmu_slice10_5 = data_5.iloc[72:95, 0:]
cmu_slice11_5 = data_5.iloc[96:119, 0:]
coop_slice1_5 = data_5.iloc[120:143, 0:]

cmu_slice3_10 = data_10.iloc[0:23, 0:]
cmu_slice4_10 = data_10.iloc[24:47, 0:]
cmu_slice6_10 = data_10.iloc[48:71, 0:]
cmu_slice10_10 = data_10.iloc[72:95, 0:]
cmu_slice11_10 = data_10.iloc[96:119, 0:]
coop_slice1_10 = data_10.iloc[120:143, 0:]

cmu_slice3_15 = data_15.iloc[0:23, 0:]
cmu_slice4_15 = data_15.iloc[24:47, 0:]
cmu_slice6_15 = data_15.iloc[48:71, 0:]
cmu_slice10_15 = data_15.iloc[72:95, 0:]
cmu_slice11_15 = data_15.iloc[96:119, 0:]
coop_slice1_15 = data_15.iloc[120:143, 0:]

cmu_slice3_20 = data_20.iloc[0:23, 0:]
cmu_slice4_20 = data_20.iloc[24:47, 0:]
cmu_slice6_20 = data_20.iloc[48:71, 0:]
cmu_slice10_20 = data_20.iloc[72:95, 0:]
cmu_slice11_20 = data_20.iloc[96:119, 0:]
coop_slice1_20 = data_20.iloc[120:143, 0:]

cmu_slice3_50 = data_50.iloc[0:23, 0:]
cmu_slice4_50 = data_50.iloc[24:47, 0:]
cmu_slice6_50 = data_50.iloc[48:71, 0:]
cmu_slice10_50 = data_50.iloc[72:95, 0:]
cmu_slice11_50 = data_50.iloc[96:119, 0:]
coop_slice1_50 = data_50.iloc[120:143, 0:]

# datasets with percentages in them
all_cmu_slice3 = [cmu_slice3_5, cmu_slice3_10, cmu_slice3_10, cmu_slice3_20, cmu_slice3_50]
all_cmu_slice4 = [cmu_slice4_5, cmu_slice4_10, cmu_slice4_10, cmu_slice4_20, cmu_slice4_50]
all_cmu_slice6 = [cmu_slice6_5, cmu_slice6_10, cmu_slice6_10, cmu_slice6_20, cmu_slice6_50]
all_cmu_slice10 = [cmu_slice10_5, cmu_slice10_10, cmu_slice10_10, cmu_slice10_20, cmu_slice10_50]
all_cmu_slice11 = [cmu_slice11_5, cmu_slice3_10, cmu_slice3_10, cmu_slice3_20, cmu_slice3_50]
all_coop_slice1 = [coop_slice1_5, coop_slice1_10, coop_slice1_10, coop_slice1_20, coop_slice1_50]

#all_datasets
datasets = [all_cmu_slice3, all_cmu_slice4, all_cmu_slice6, all_cmu_slice10, all_cmu_slice11, all_coop_slice1]

# use for later, graphs
cmu_only_datasets = [all_cmu_slice3, all_cmu_slice4, all_cmu_slice6, all_cmu_slice10, all_cmu_slice11]
coop_only_dataset = [all_coop_slice1]

print("Applying Z-Transform")

all_slices_str = ["all_cmu_slice3", "all_cmu_slice4", "all_cmu_slice6", "all_cmu_slice10", "all_cmu_slice11", "all_coop_slice1"]
all_slices_str_readable = ["CMU Slice3", "CMU Slice4", "CMU Slice6", "CMU Slice10", "CMU Slice11", "Retail Shop"]
percentages_str = ["5%", "10%", "15%", "20%", "50%"]

csv_results_arr = np.array(["Dataset", "5%", "10%", "15%", "20%", "50%"])

# comment out which on you want to use
# model_indices = np.delete(np.arange(1,23), [1,21,20]) #remove the classifier trained on all, baseline - do not depend on percentages - DOUBLE check indices
model_indices = np.arange(1,23) #all models

all_slices_str_idx = 0
for dataset in datasets:
    print("Dataset: " + all_slices_str[all_slices_str_idx])
    percentages_str_idx = 0
    percentages_results_arr = np.array([all_slices_str[all_slices_str_idx]])

    for percentage in dataset:
        conc_times_all_ml_methods = percentage.iloc[model_indices, 4]
        fm_times_all_ml_methods = percentage.iloc[model_indices, 5]
        t_err_all_ml_methods = percentage.iloc[model_indices, 7]
        rot_err_all_ml_methods = percentage.iloc[model_indices, 8]

        # conc_times_all_ml_methods_z_transformed = (conc_times_all_ml_methods - conc_times_all_ml_methods.mean() ) / conc_times_all_ml_methods.std()
        fm_times_all_ml_methods_z_transformed = (fm_times_all_ml_methods - fm_times_all_ml_methods.mean() ) / fm_times_all_ml_methods.std()
        t_err_all_ml_methods_z_transformed = (t_err_all_ml_methods - t_err_all_ml_methods.mean() ) / t_err_all_ml_methods.std()
        rot_err_ml_methods_z_transformed = (rot_err_all_ml_methods - rot_err_all_ml_methods.mean() ) / rot_err_all_ml_methods.std()

        all_metrics_ml_methods_z_transformed = pd.concat([#conc_times_all_ml_methods_z_transformed,
                                                          fm_times_all_ml_methods_z_transformed,
                                                          t_err_all_ml_methods_z_transformed,
                                                          rot_err_ml_methods_z_transformed], axis=1)

        min_idx = all_metrics_ml_methods_z_transformed.mean(axis=1).rank(ascending=True).idxmin()
        best_method = percentage['method'][min_idx]

        print(" percentage: " + percentages_str[percentages_str_idx])
        print("  best method: " + best_method)
        percentages_str_idx += 1
        percentages_results_arr = np.append(percentages_results_arr, best_method)

    csv_results_arr = np.vstack((csv_results_arr, percentages_results_arr))
    all_slices_str_idx += 1
    print()

np.savetxt("plots/best_methods_per_dataset.csv" , csv_results_arr , delimiter="," , fmt='%s')

best_model_each_dataset = {}
for i in range(len(csv_results_arr)-1):
    i += 1
    unique, pos = np.unique(csv_results_arr[i,1:],return_inverse=True)
    counts = np.bincount(pos)
    maxpos = counts.argmax()
    best_model_each_dataset[csv_results_arr[i,0]] = unique[maxpos]

# for Coop
for k,v in best_model_each_dataset.items():
    if k == 'all_coop_slice1':
        coop_best_performing_method = v

# for CMU datasets or slices
cmu_best_methods = []
for k,v in best_model_each_dataset.items():
    if k != 'all_coop_slice1':
        cmu_best_methods.append(v)

unique, pos = np.unique(cmu_best_methods, return_inverse=True)
counts = np.bincount(pos)
maxpos = counts.argmax()
cmu_all_best_performing_method = unique[maxpos]

baseline_method = 'All (~800)'
print(f"{coop_best_performing_method=}")
print(f"{cmu_all_best_performing_method=}")

print("Saving Graphs..")

# Plotting starts here
plt.figure(figsize=(11,6), dpi=100)

print("Bar Charts..")
plt.figure(5)
all_slices_str_idx = 0
fm_times_best_model = np.array([])
t_err_best_model = np.array([])
rot_err_best_model = np.array([])
fm_times_baseline = np.array([])
t_err_baseline = np.array([])
rot_err_baseline = np.array([])

for dataset in datasets:
    dataset_name = all_slices_str[all_slices_str_idx]
    print(" Graphs for " + dataset_name + " best model only")

    # these will be the same for each percentage - because baseline, and classifier all (best model) always uses all the features
    first_percentage = dataset[0] #doesnt matter which one here (0 is 5%)

    fm_times_baseline = np.append(fm_times_baseline, first_percentage.loc[first_percentage['method'] == baseline_method]['fm_time'] * 1000)  # convert to milliseconds
    t_err_baseline = np.append(t_err_baseline, first_percentage.loc[first_percentage['method'] == baseline_method]['t_err'])
    rot_err_baseline = np.append(rot_err_baseline, first_percentage.loc[first_percentage['method'] == baseline_method]['rot_err'])

    if dataset_name == 'all_coop_slice1':
        best_method = coop_best_performing_method
    else:
        best_method = cmu_all_best_performing_method

    fm_times_best_model = np.append(fm_times_best_model, first_percentage.loc[first_percentage['method'] == best_method]['fm_time'] * 1000)  # convert to milliseconds
    t_err_best_model = np.append(t_err_best_model, first_percentage.loc[first_percentage['method'] == best_method]['t_err'])
    rot_err_best_model = np.append(rot_err_best_model, first_percentage.loc[first_percentage['method'] == best_method]['rot_err'])

plt.cla()
ind = np.arange(0,3*len(fm_times_baseline[0:5]), 3)
width = 0.8
plt.bar(ind, fm_times_baseline[0:5], width, label='Feature Matching Tme (Baseline)')
plt.bar(ind + width, fm_times_best_model[0:5], width, label='Feature Matching Tme (Best model)')

plt.ylabel('Time (ms)', fontsize=20)

plt.xticks(ind + width/2, (all_slices_str_readable[0:-1]))
plt.legend(loc='upper center', framealpha=1, fontsize=16, shadow = True)
plt.savefig("plots/"+dataset_name+"_fm_times_all_cmu_bar.pdf")

import pdb
pdb.set_trace()

# for seperate slices figures.
# CMU
plt.figure(0)
all_slices_str_idx = 0
for dataset in cmu_only_datasets:
    dataset_name = all_slices_str[all_slices_str_idx]
    print(" Graphs for " + dataset_name + " best model only")
    fm_times = np.array([])
    t_err = np.array([])
    rot_err = np.array([])
    fm_times_baseline = np.array([])
    t_err_baseline = np.array([])
    rot_err_baseline = np.array([])

    # these will be the same for each percentage - because baseline always uses all the features
    first_percentage = dataset[0] #doesnt matter which one here (0 is 5%)
    fm_times_baseline = np.append(fm_times_baseline, first_percentage.loc[first_percentage['method'] == baseline_method]['fm_time'] * 1000)  # convert to milliseconds
    t_err_baseline = np.append(t_err_baseline, first_percentage.loc[first_percentage['method'] == baseline_method]['t_err'])
    rot_err_baseline = np.append(rot_err_baseline, first_percentage.loc[first_percentage['method'] == baseline_method]['rot_err'])
    fm_times_baseline = np.repeat(fm_times_baseline, 5)
    t_err_baseline = np.repeat(t_err_baseline, 5)
    rot_err_baseline = np.repeat(rot_err_baseline, 5)

    for percentage in dataset:
        fm_times = np.append(fm_times, percentage.loc[percentage['method'] == cmu_all_best_performing_method]['fm_time'] * 1000) #convert to milliseconds
        t_err = np.append(t_err, percentage.loc[percentage['method'] == cmu_all_best_performing_method]['t_err'])
        rot_err = np.append(rot_err, percentage.loc[percentage['method'] == cmu_all_best_performing_method]['rot_err'])

    percentage = [5, 10, 15, 20, 50]
    x2 = np.arange(len(percentage))

    plt.cla()
    plt.plot(x2, fm_times, label="Best Model")
    plt.plot(x2, fm_times_baseline, label="Baseline Model")
    plt.title('Baseline VS Best Performing Model for ' + all_slices_str_readable[all_slices_str_idx])
    plt.xlabel('Percentage %')
    plt.ylabel('Time (ms)')
    plt.xticks(np.arange(5), percentage)
    plt.legend(loc='best', framealpha=1,  shadow=True)
    plt.savefig("plots/"+dataset_name+"_fm_times_per_percentage.pdf")

    plt.cla()
    plt.plot(x2, t_err, label="Best Model")
    plt.plot(x2, t_err_baseline, label="Baseline Model")
    plt.title('Baseline VS Best Performing Model for CMU datasets ' + all_slices_str_readable[all_slices_str_idx])
    plt.xlabel('Percentage %')
    plt.ylabel('Translation Error (m)')
    plt.xticks(np.arange(5), percentage)
    plt.legend(loc='best', framealpha=1,  shadow=True)
    plt.savefig("plots/"+dataset_name+"_t_error_per_percentage.pdf")

    plt.cla()
    plt.plot(x2, rot_err, label="Best Model")
    plt.plot(x2, rot_err_baseline, label="Baseline Model")
    plt.title('Baseline VS Best Performing Model for CMU datasets ' + all_slices_str_readable[all_slices_str_idx])
    plt.xlabel('Percentage %')
    plt.ylabel('Rotation Error in degrees')
    plt.xticks(np.arange(5), percentage)
    plt.legend(loc='best', framealpha=1,  shadow=True)
    plt.savefig("plots/"+dataset_name+"_rot_err_per_percentage.pdf")

    all_slices_str_idx += 1

print()

# Coop
plt.figure(1)
coop_only_dataset = coop_only_dataset[0] #get the first one
dataset_name = all_slices_str[-1]
print(" Graphs for " + dataset_name  + " best model only")
fm_times = np.array([])
t_err = np.array([])
rot_err = np.array([])
fm_times_baseline = np.array([])
t_err_baseline = np.array([])
rot_err_baseline = np.array([])

# these will be the same for each percentage - because baseline always uses all the features
first_percentage = coop_only_dataset[0]  # doesnt matter which one here (0 is 5%)
fm_times_baseline = np.append(fm_times_baseline, first_percentage.loc[first_percentage['method'] == baseline_method]['fm_time'] * 1000)  # convert to milliseconds
t_err_baseline = np.append(t_err_baseline, first_percentage.loc[first_percentage['method'] == baseline_method]['t_err'])
rot_err_baseline = np.append(rot_err_baseline, first_percentage.loc[first_percentage['method'] == baseline_method]['rot_err'])
fm_times_baseline = np.repeat(fm_times_baseline, 5)
t_err_baseline = np.repeat(t_err_baseline, 5)
rot_err_baseline = np.repeat(rot_err_baseline, 5)

for percentage in coop_only_dataset:
    fm_times = np.append(fm_times, percentage.loc[percentage['method'] == coop_best_performing_method]['fm_time'] * 1000) #convert to milliseconds
    t_err = np.append(t_err, percentage.loc[percentage['method'] == coop_best_performing_method]['t_err'])
    rot_err = np.append(rot_err, percentage.loc[percentage['method'] == coop_best_performing_method]['rot_err'])

percentage = [5, 10, 15, 20, 50]
x2 = np.arange(len(percentage))

plt.cla()
plt.plot(x2, fm_times, label="Best Model")
plt.plot(x2, fm_times_baseline, label="Baseline Model")
plt.title('Baseline VS Best Performing Model for ' + all_slices_str_readable[-1])
plt.xlabel('Percentage %')
plt.ylabel('Time (ms)')
plt.xticks(np.arange(5), percentage)
plt.legend(loc='best', framealpha=1,  shadow=True)
plt.savefig("plots/"+dataset_name+"_fm_times_per_percentage.pdf")

plt.cla()
plt.plot(x2, t_err, label="Best Model")
plt.plot(x2, t_err_baseline, label="Baseline Model")
plt.title('Baseline VS Best Performing Model for ' + all_slices_str_readable[-1])
plt.xlabel('Percentage %')
plt.ylabel('Translation Error (m)')
plt.xticks(np.arange(5), percentage)
plt.legend(loc='best', framealpha=1,  shadow=True)
plt.savefig("plots/"+dataset_name+"_t_error_per_percentage.pdf")

plt.cla()
plt.plot(x2, rot_err, label="Best Model")
plt.plot(x2, rot_err_baseline, label="Baseline Model")
plt.title('Baseline VS Best Performing Model for ' + all_slices_str_readable[-1])
plt.xlabel('Percentage %')
plt.ylabel('Rotation Error in degrees')
plt.xticks(np.arange(5), percentage)
plt.legend(loc='best', framealpha=1,  shadow=True)
plt.savefig("plots/"+dataset_name+"_rot_err_per_percentage.pdf")

print()

# for 1 figure per error, time etc (duplicate code as above)
all_slices_str_idx = 0
for dataset in cmu_only_datasets:
    dataset_name = all_slices_str[all_slices_str_idx]
    print(" Graphs for " + dataset_name + " accumulating")
    fm_times = np.array([])
    t_err = np.array([])
    rot_err = np.array([])
    fm_times_baseline = np.array([])
    t_err_baseline = np.array([])
    rot_err_baseline = np.array([])

    # these will be the same for each percentage - because baseline always uses all the features
    first_percentage = dataset[0] #doesnt matter which one here
    fm_times_baseline = np.append(fm_times_baseline, first_percentage.loc[first_percentage['method'] == baseline_method]['fm_time'] * 1000)  # convert to milliseconds
    t_err_baseline = np.append(t_err_baseline, first_percentage.loc[first_percentage['method'] == baseline_method]['t_err'])
    rot_err_baseline = np.append(rot_err_baseline, first_percentage.loc[first_percentage['method'] == baseline_method]['rot_err'])
    fm_times_baseline = np.repeat(fm_times_baseline, 5)
    t_err_baseline = np.repeat(t_err_baseline, 5)
    rot_err_baseline = np.repeat(rot_err_baseline, 5)

    for percentage in dataset:
        fm_times = np.append(fm_times, percentage.loc[percentage['method'] == cmu_all_best_performing_method]['fm_time'] * 1000) #convert to milliseconds
        t_err = np.append(t_err, percentage.loc[percentage['method'] == cmu_all_best_performing_method]['t_err'])
        rot_err = np.append(rot_err, percentage.loc[percentage['method'] == cmu_all_best_performing_method]['rot_err'])

    percentage = [5, 10, 15, 20, 50]
    x2 = np.arange(len(percentage))

    plt.figure(2)
    plt.plot(x2, fm_times, label="Best Model for " + all_slices_str_readable[all_slices_str_idx])
    if(all_slices_str_idx==0): #only add this once because it is baseline!
        plt.plot(x2, fm_times_baseline, label="Baseline Model")
    plt.title('Baseline VS Best Performing Models for CMU', fontsize=10)
    plt.xlabel('Percentage %', fontsize=10)
    plt.ylabel('Time (ms)', fontsize=10)
    plt.xticks(np.arange(5), percentage)
    plt.legend(loc='best', framealpha=1,  shadow=True, fontsize=8)
    plt.savefig("plots/fm_times_per_percentage_accumulating_all_cmu.pdf")

    plt.figure(3)
    plt.plot(x2, t_err, label="Best Model for " + all_slices_str_readable[all_slices_str_idx])
    if (all_slices_str_idx == 0):  # only add this once because it is baseline!
        plt.plot(x2, t_err_baseline, label="Baseline Model")
    plt.title('Baseline VS Best Performing Model for CMU', fontsize=10)
    plt.xlabel('Percentage %', fontsize=10)
    plt.ylabel('Translation Error (m)', fontsize=10)
    plt.xticks(np.arange(5), percentage)
    plt.legend(loc='best', framealpha=1,  shadow=True, fontsize=8)
    plt.savefig("plots/t_error_per_percentage_accumulating_all_cmu.pdf")

    plt.figure(4)
    plt.plot(x2, rot_err, label="Best Model for " + all_slices_str_readable[all_slices_str_idx])
    if (all_slices_str_idx == 0):  # only add this once because it is baseline!
        plt.plot(x2, rot_err_baseline, label="Baseline Model")
    plt.title('Baseline VS Best Performing Model for CMU', fontsize=10)
    plt.xlabel('Percentage %', fontsize=10)
    plt.ylabel('Rotation Error in degrees', fontsize=10)
    plt.xticks(np.arange(5), percentage)
    plt.legend(loc='best', framealpha=1,  shadow=True, fontsize=8)
    plt.savefig("plots/rot_err_per_percentage_accumulating_all_cmu.pdf")

    all_slices_str_idx += 1

exit()

# Old code here

plt.figure(figsize=(11,4), dpi=100)

N = 7
inliers_0 = (93 , 95 , 98 , 100 , 101 , 102 , 104)
inliers_1 = (81 , 83 , 83 , 85 , 88 , 89 , 89 )

ind = np.arange(0,3*N, 3)
width = 0.8
plt.bar(ind, inliers_0, width, label='Inliers for vanilla RANSAC')
plt.bar(ind + width, inliers_1, width, label='Inliers for PROSAC (lowe\'s ratio by reliability score ratio)')

plt.ylabel('Inliers - Live Model', fontsize=20)
# plt.title('Inliers for vanilla RANSAC and PROSAC against live model', fontsize=20)

plt.xticks(ind + width/2, ('+s1', '+s2', '+s3', '+s4', '+s5', '+s6', '+s7'))
plt.legend(loc='upper center', framealpha=1, fontsize=16, shadow = True)

ax = plt.gca()
ax.tick_params(axis="y", labelsize=18)
ax.tick_params(axis="x", labelsize=18)
ax.set_ylim([60,126])

plt.savefig('/Users/alex/Projects/EngDLocalProjects/Papers Local - Before iCloud/paper/figures/inliers_figure.pdf')

# ---------------------------

# This is for vanillia RANSAC - Live model
fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Values for vanilla RANSAC against live model')

xlabels = ['Base', '+s1', '+s2', '+s3', '+s4', '+s5', '+s6', '+s7']
time = (0.41, 0.22 , 0.19 , 0.17 , 0.12 , 0.14 , 0.14 , 0.12)
trans_err = (0.07, 0.04 , 0.03 , 0.03 , 0.03 , 0.02 , 0.02 , 0.02)
rot_err = np.array([5.01, 3.61 , 3.62 , 3.29 , 2.75 , 2.43 , 2.41 , 2.18])

# This is for PROSAC version that uses the inverse lowe's ratio by the reliability score ratio - Live model
# plt.figure(figsize=(11,4), dpi=100)
# N = 7
# time = (0.09 , 0.09 , 0.09 , 0.05 , 0.07 , 0.06 , 0.05)
# trans_err = (0.05 , 0.05 , 0.05 , 0.04 , 0.04 , 0.04 , 0.04)
# rot_err = np.array([4.68 , 4.64 , 4.87 , 4.13 , 4.06 , 3.92 , 3.79])

ax1.bar(xlabels, time, color = 'slateblue')
ax1.legend(['Time'], framealpha=1, shadow = True)
ax1.set_ylabel('Seconds')
ax1.set_xticklabels(xlabels)

ax2.bar(xlabels, trans_err, color = 'crimson')
ax2.legend(['Translation Error'], framealpha=1, shadow = True)
ax2.set_ylabel('Meters')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.set_xticklabels(xlabels)

ax3.bar(xlabels, rot_err, color = 'seagreen')
ax3.legend(['Rotation Error'], framealpha=1, shadow = True)
ax3.set_ylabel('Degrees')
ax3.set_xticklabels(xlabels)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/Users/alex/Projects/EngDLocalProjects/Papers Local - Before iCloud/paper/figures/benchmark_values_ransac.pdf')

# ---------------------------

# This is for PROSAC version that uses the inverse lowe's ratio by the reliability score ratio - Live model
fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Values for PROSAC against live model')

xlabels = ['Base', '+s1', '+s2', '+s3', '+s4', '+s5', '+s6', '+s7']
time = (0.10, 0.09 , 0.09 , 0.09 , 0.05 , 0.07 , 0.06 , 0.05)
trans_err = (0.08, 0.05 , 0.05 , 0.05 , 0.04 , 0.04 , 0.04 , 0.04)
rot_err = np.array([7.39, 4.68 , 4.64 , 4.87 , 4.13 , 4.06 , 3.92 , 3.79])

ax1.bar(xlabels, time, color = 'slateblue')
ax1.legend(['Time'], framealpha=1, shadow = True)
ax1.set_ylabel('Seconds')
ax1.set_xticklabels(xlabels)

ax2.bar(xlabels, trans_err, color = 'crimson')
ax2.legend(['Translation Error'], framealpha=1, shadow = True)
ax2.set_ylabel('Meters')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.set_xticklabels(xlabels)

ax3.bar(xlabels, rot_err, color = 'seagreen')
ax3.legend(['Rotation Error'], framealpha=1, shadow = True)
ax3.set_ylabel('Degrees')
ax3.set_xticklabels(xlabels)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('/Users/alex/Projects/EngDLocalProjects/Papers Local - Before iCloud/paper/figures/benchmark_values_prosac.pdf')

