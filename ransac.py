# This is vanilla RANSAC Implementation and Modified one
import numpy as np
import cv2
import time

# intrinsics matrix
K = np.loadtxt(
        "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/matrices/pixel_intrinsics_low_640_portrait.txt")

def run_ransac(matches_for_image):
    s = 4  # or minimal_sample_size
    p = 0.99 # this is a typical value
    # number of iterations (http://www.cse.psu.edu/~rtc12/CSE486/lecture15.pdf and https://youtu.be/5E5n7fhLHEM?list=PLTBdjV_4f-EKeki5ps2WHqJqyQvxls4ha&t=428)
    # also reddit post https://www.reddit.com/r/computervision/comments/gikj1s/can_somebody_help_me_understand_ransac/
    no_iterations = 1000  # can set this to whatever you want to start with
    k = 0
    distCoeffs = np.zeros((5, 1))  # assume zero for now
    threshold = 8.0 # same as opencv
    max = np.iinfo(np.int32).min
    best_model = {}
    elapsed_time_total_for_random_sampling = 0
    while k < no_iterations:
        inliers = []
        # pick 4 random matches (assume they are inliers)
        start = time.time()
        random_matches = np.random.choice(len(matches_for_image), s, replace=False)
        end = time.time()
        elapsed_time = end - start
        elapsed_time_total_for_random_sampling = elapsed_time_total_for_random_sampling + elapsed_time
        # get 3D and 2D points
        obj_points = matches_for_image[(random_matches), 2:5]
        img_points = matches_for_image[(random_matches), 0:2]

        # calculate pose
        retval, rvec, tvec = cv2.solvePnP(obj_points,img_points,K,distCoeffs)
        rotm = cv2.Rodrigues(rvec)[0]
        Rt = np.r_[(np.c_[rotm, tvec]), [np.array([0, 0, 0, 1])]]

        # run against all the other matches (except the ones you already picked)
        for i in range(len(matches_for_image)):
            if(i not in random_matches):
                obj_point = matches_for_image[i, 2:5]
                img_point_gt = matches_for_image[i, 0:2]
                obj_point = np.r_[obj_point, 1] #make homogeneous
                img_point_est = K.dot(Rt.dot(obj_point.transpose())[0:3])
                img_point_est = img_point_est / img_point_est[2] #divide by last coordinate
                dist = np.linalg.norm(img_point_gt - img_point_est[0:2])
                if(dist < threshold):
                    inliers.append(matches_for_image[i])

        inlers_no = len(inliers) + s #total number of inliers
        outliers_no = len(matches_for_image) - inlers_no

        # store best model so far
        if(inlers_no > max):
            best_model['Rt'] = Rt
            best_model['inlers_no'] = inlers_no
            max = inlers_no
            e = outliers_no / len(matches_for_image)
            N = np.log(1 - p) / np.log(1 - np.power((1 - e), s))
            N = int(np.floor(N))
            no_iterations = N
            if(k > N): # this is saying if the max number of iterations you should have run is N, but you already did k > N then no point continuing
                return (inlers_no, outliers_no, k, best_model, elapsed_time_total_for_random_sampling)

        k = k + 1

    return (inlers_no, outliers_no, k, best_model, elapsed_time_total_for_random_sampling)

def run_ransac_modified(matches_for_image, distribution):
    s = 4  # or minimal_sample_size
    p = 0.99 # this is a typical value
    # number of iterations (http://www.cse.psu.edu/~rtc12/CSE486/lecture15.pdf and https://youtu.be/5E5n7fhLHEM?list=PLTBdjV_4f-EKeki5ps2WHqJqyQvxls4ha&t=428)
    # also reddit post https://www.reddit.com/r/computervision/comments/gikj1s/can_somebody_help_me_understand_ransac/
    no_iterations = 1000  # can set this to whatever you want to start with
    k = 0
    distCoeffs = np.zeros((5, 1))  # assume zero for now
    threshold = 8.0 # same as opencv
    max = np.iinfo(np.int32).min
    best_model = {}
    elapsed_time_total_for_random_sampling = 0
    while k < no_iterations:
        inliers = []
        # pick 4 random matches (assume they are inliers)
        start = time.time()
        random_matches = np.random.choice(len(matches_for_image), s , p = distribution, replace=False)
        end = time.time()
        elapsed_time = end - start
        elapsed_time_total_for_random_sampling = elapsed_time_total_for_random_sampling + elapsed_time
        # get 3D and 2D points
        obj_points = matches_for_image[(random_matches), 2:5]
        img_points = matches_for_image[(random_matches), 0:2]

        # calculate pose
        retval, rvec, tvec = cv2.solvePnP(obj_points,img_points,K,distCoeffs)
        rotm = cv2.Rodrigues(rvec)[0]
        Rt = np.r_[(np.c_[rotm, tvec]), [np.array([0, 0, 0, 1])]]

        # run against all the other matches (except the ones you already picked)
        for i in range(len(matches_for_image)):
            if(i not in random_matches):
                obj_point = matches_for_image[i, 2:5]
                img_point_gt = matches_for_image[i, 0:2]
                obj_point = np.r_[obj_point, 1] #make homogeneous
                img_point_est = K.dot(Rt.dot(obj_point.transpose())[0:3])
                img_point_est = img_point_est / img_point_est[2] #divide by last coordinate
                dist = np.linalg.norm(img_point_gt - img_point_est[0:2])
                if(dist < threshold):
                    inliers.append(matches_for_image[i])

        inlers_no = len(inliers) + s #total number of inliers
        outliers_no = len(matches_for_image) - inlers_no

        # store best model so far
        if(inlers_no > max):
            best_model['Rt'] = Rt
            best_model['inlers_no'] = inlers_no
            max = inlers_no
            e = outliers_no / len(matches_for_image)
            N = np.log(1 - p) / np.log(1 - np.power((1 - e), s))
            N = int(np.floor(N))
            no_iterations = N
            if(k > N): # this is saying if the max number of iterations you should have run is N, but you already did k > N then no point continuing
                return (inlers_no, outliers_no, k, best_model, elapsed_time_total_for_random_sampling)

        k = k + 1

    return (inlers_no, outliers_no, k, best_model, elapsed_time_total_for_random_sampling)