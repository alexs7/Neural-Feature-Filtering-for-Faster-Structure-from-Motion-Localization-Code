import os

cmu_slice3 = "colmap_data/CMU_data/slice3/ML_data/results_evaluator.csv"
cmu_slice4 = "colmap_data/CMU_data/slice4/ML_data/results_evaluator.csv"
cmu_slice6 = "colmap_data/CMU_data/slice6/ML_data/results_evaluator.csv"
cmu_slice10 = "colmap_data/CMU_data/slice10/ML_data/results_evaluator.csv"
cmu_slice11 = "colmap_data/CMU_data/slice11/ML_data/results_evaluator.csv"
coop_slice1 = "colmap_data/Coop_data/slice1/ML_data/results_evaluator.csv"

results  = [cmu_slice3, cmu_slice4, cmu_slice6, cmu_slice10, cmu_slice11, coop_slice1]

for result in results:
    os.system("cat " + result)
    print()
    print()