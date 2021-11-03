# This is for the ML part, the second paper
# run this on weatherwax (ssd fast1)
import sys
import numpy as np

image_pose_errors_cmu_3 = np.load("colmap_data/CMU_data/slice3/ML_data/image_pose_errors_all_10.npy", allow_pickle=True)
image_pose_errors_cmu_4 = np.load("colmap_data/CMU_data/slice4/ML_data/image_pose_errors_all_10.npy", allow_pickle=True)
image_pose_errors_cmu_6 = np.load("colmap_data/CMU_data/slice6/ML_data/image_pose_errors_all_10.npy", allow_pickle=True)
image_pose_errors_cmu_10 = np.load("colmap_data/CMU_data/slice10/ML_data/image_pose_errors_all_10.npy", allow_pickle=True)
image_pose_errors_cmu_11 = np.load("colmap_data/CMU_data/slice11/ML_data/image_pose_errors_all_10.npy", allow_pickle=True)
image_pose_errors_coop_1 = np.load("colmap_data/CMU_data/slice11/ML_data/image_pose_errors_all_10.npy", allow_pickle=True)

# CMU
results_cmu_titles = ["--slice3", "--slice4", "--slice6", "--slice10", "--slice11"]
results_cmu = [image_pose_errors_cmu_3, image_pose_errors_cmu_4, image_pose_errors_cmu_6, image_pose_errors_cmu_10, image_pose_errors_cmu_11]
results_coop = image_pose_errors_coop_1 # just for naming

# Print options
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

# all the cases tried in, model_evaluator.py. The order is the same in image_pose_errors_all_*.npy files too
method_names_array = [
    "Classifier w/ top 10% matches",
    "Classifier using all matches",
    "Classifier and Regressor w/ score per image",
    "Classifier and Regressor w/ score per session",
    "Classifier and Regressor w/ visibility score",
    "Regressor w/ score per image",
    "Regressor w/ score per session",
    "Regressor w/ visibility score",
    "Combined w/ score per image",
    "Combined w/ score per session",
    "Combined w/ visibility score",
    "Class. and Regr. w/ score per image, dist. RANSAC",
    "Class. and Regr. w/ score per session, dist. RANSAC",
    "Class. and Regr. w/ visibility score, dist. RANSAC",
    "Regressor w/ score per image, PROSAC",
    "Regressor w/ score per session, PROSAC",
    "Regressor w/ visibility score, PROSAC",
    "Combined w/ score per image, PROSAC",
    "Combined w/ score per session, PROSAC",
    "Combined w/ visibility score, PROSAC",
    "Random feature case",
    "Baseline using all features"
]

#  for CMU
for k in range(len(results_cmu)):
    result = results_cmu[k] #result for each slice (i.e. the errors per method for all query images)
    slice_title = results_cmu_titles[k]
    print(slice_title)
    print("high | medium | coarse - Bucket approach")

    # result and method_names_array are the same size and order as in model_evaluator.py.
    for i in range(len(result)):
        print()
        bucket_high = 0  # 0.25m, 2d
        bucket_medium = 0  # 0.5m, 5d
        bucket_coarse = 0  # 5m, 10d

        print(slice_title + " " + method_names_array[i])

        total_images = len(result[i]) # result[i] - is all the images for that slice
        for image_name , errors in result[i].items():
            t_error = errors[0]
            r_error = errors[1]
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

        print(" %2.1f | %2.1f | %2.1f " %(bucket_high_percentage, bucket_medium_percentage ,bucket_coarse_percentage) )

#  for coop
print("Coop Results")
print("high | medium | coarse - Bucket approach")

# result and method_names_array are the same size and order as in model_evaluator.py.
for i in range(len(results_coop)):
    print()
    bucket_high = 0  # 0.25m, 2d
    bucket_medium = 0  # 0.5m, 5d
    bucket_coarse = 0  # 5m, 10d

    print("--Coop " + method_names_array[i])

    total_images = len(results_coop[i]) # result[i] - is all the images for that slice
    for image_name , errors in results_coop[i].items():
        t_error = errors[0]
        r_error = errors[1]
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

    print(" %2.1f | %2.1f | %2.1f " %(bucket_high_percentage, bucket_medium_percentage ,bucket_coarse_percentage) )