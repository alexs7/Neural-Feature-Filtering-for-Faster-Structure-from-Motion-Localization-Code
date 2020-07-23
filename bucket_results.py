import sys
import numpy as np

path = sys.argv[1]
results = np.load(path,allow_pickle=True)

# Print options
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

def create_buckets(results):
    print("high | medium | coarse")
    for k,v in results.item().items():
        bucket_high = 0  # 0.25m, 2d
        bucket_medium = 0  # 0.5m, 5d
        bucket_coarse = 0  # 5m, 10d

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

        bucket_high_percentage = 100 * bucket_high / total_images
        bucket_medium_percentage = 100 * bucket_medium / total_images
        bucket_coarse_percentage = 100 * bucket_coarse / total_images

        print(k + " %2.2f | %2.2f | %2.2f " %(bucket_high_percentage, bucket_medium_percentage ,bucket_coarse_percentage) )

create_buckets(results)