import sys
import numpy as np

result_3 = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/slice3/results.npy", allow_pickle=True).item()
result_4 = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/slice4/results.npy", allow_pickle=True).item()
result_6 = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/slice6/results.npy", allow_pickle=True).item()
result_10 = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/slice10/results.npy", allow_pickle=True).item()
result_11 = np.load("/home/alex/fullpipeline/colmap_data/CMU_data/slice11/results.npy", allow_pickle=True).item()

# CMU
results_cmu = [result_3, result_4, result_6, result_10, result_11]

# Print options
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

# For paper
ransac_types = ['ransac_base', 'prosac_base', 'ransac_live', 'ransac_dist_heatmap_val',
                'ransac_dist_visibility_score', 'inverse_lowes_ratio', 'reliability_higher_neighbour_heatmap_value', 'reliability_higher_neighbour_score',
                'higher_neighbour_visibility_score', 'lowes_by_reliability_score_ratio']

print("RANSAC Types number: " + str(len(ransac_types)))

def create_buckets(results):
    for result in results:
        print("high | medium | coarse")
        for k,v in result.items():

            if(k in ransac_types):
                bucket_high = 0  # 0.25m, 2d
                bucket_medium = 0  # 0.5m, 5d
                bucket_coarse = 0  # 5m, 10d

                import pdb; pdb.set_trace()
                total_images = len(v[2])
                trans_errors = v[2]
                rot_errors = v[3]
                for i in range(len(trans_errors)): #can be 3 too
                    t_error = trans_errors[i]
                    r_error = rot_errors[i]
                    if(t_error < 0.25 and r_error < 2):
                        bucket_high += 1
                    if(t_error < 0.5 and r_error < 5):
                        bucket_medium += 1
                    if(t_error < 5 and r_error < 10):
                        bucket_coarse += 1

                print()
                print(' total_images: ' + str(total_images))
                print(' bucket_high: ' + str(bucket_high))
                print(' bucket_medium: ' + str(bucket_medium))
                print(' bucket_coarse: ' + str(bucket_coarse))
                bucket_high_percentage = 100 * bucket_high / total_images
                bucket_medium_percentage = 100 * bucket_medium / total_images
                bucket_coarse_percentage = 100 * bucket_coarse / total_images

                print(k + " %2.2f | %2.2f | %2.2f " %(bucket_high_percentage, bucket_medium_percentage ,bucket_coarse_percentage) )

create_buckets(results_cmu)
