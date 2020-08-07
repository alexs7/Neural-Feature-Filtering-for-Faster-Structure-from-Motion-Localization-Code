import numpy as np

result_2 = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/slice2/results.npy", allow_pickle=True).item()
result_3 = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/slice3/results.npy", allow_pickle=True).item()
result_4 = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/slice4/results.npy", allow_pickle=True).item()
result_6 = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/slice6/results.npy", allow_pickle=True).item()
result_10 = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/slice10/results.npy", allow_pickle=True).item()
result_10 = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/slice11/results.npy", allow_pickle=True).item()
result_coop = np.load("/home/alex/fullpipeline/colmap_data/Coop_data/slice1/results.npy", allow_pickle=True).item()

results = [result_2, result_3, result_4, result_6, result_10, result_coop]

for result in results:
    
