# This is vanilla RANSAC Implementation and Modified one
import numpy as np
import cv2

MAX_RANSAC_ITERS = 5000
# intrinsics matrix
K = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/matrices/pixel_intrinsics_low_640_portrait.txt")

def get_pose(img_points, obj_points):
    distCoeffs = np.zeros((5, 1))
    # calculate pose
    # this is required for SOLVEPNP_P3P
    img_points = np.ascontiguousarray(img_points[:, :2]).reshape((img_points.shape[0], 1, 2))
    retval, rvec, tvec = cv2.solvePnP(obj_points.astype(np.float32), img_points.astype(np.float32), K, distCoeffs, flags=cv2.SOLVEPNP_P3P)
    rotm = cv2.Rodrigues(rvec)[0]
    Rt = np.r_[(np.c_[rotm, tvec]), [np.array([0, 0, 0, 1])]]
    return Rt

def model_fit(matches_for_image, Rt, threshold):
    obj_point = matches_for_image[:, 2:5]
    img_point_gt = matches_for_image[:, 0:2]
    obj_point = np.hstack((obj_point, np.ones((obj_point.shape[0],1)))) # make homogeneous
    img_point_est = K.dot(Rt.dot(obj_point.transpose())[0:3])
    img_point_est = img_point_est / img_point_est[2]  # divide by last coordinate
    img_point_est = img_point_est.transpose()
    dist = np.linalg.norm(img_point_gt[:,0:2] - img_point_est[:,0:2], axis=-1)
    inliers = matches_for_image[dist < threshold, :]
    inliers_indices = inliers.nonzero()
    return inliers, inliers_indices

def ransac(matches_for_image):
    s = 4  # or minimal_sample_size
    p = 0.99 # this is a typical value
    # number of iterations (http://www.cse.psu.edu/~rtc12/CSE486/lecture15.pdf and https://youtu.be/5E5n7fhLHEM?list=PLTBdjV_4f-EKeki5ps2WHqJqyQvxls4ha&t=428)
    # also reddit post https://www.reddit.com/r/computervision/comments/gikj1s/can_somebody_help_me_understand_ransac/
    no_iterations = MAX_RANSAC_ITERS  # can set this to whatever you want to start with
    k = 0
    threshold = 8.0 # 8.0 same as opencv
    max = np.iinfo(np.int32).min
    best_model = {}

    while k < no_iterations:
        k = k + 1
        # pick 4 random matches (assume they are inliers)
        random_matches = np.random.choice(len(matches_for_image), s, replace=False)

        # get 3D and 2D points
        obj_points = matches_for_image[(random_matches), 2:5]
        img_points = matches_for_image[(random_matches), 0:2]

        Rt = get_pose(img_points, obj_points)
        inliers, _ = model_fit(matches_for_image, Rt, threshold)

        inlers_no = len(inliers) + s #total number of inliers
        outliers_no = len(matches_for_image) - inlers_no

        # store best model so far
        if(inlers_no > max):
            best_model['Rt'] = Rt
            best_model['inlers_no'] = inlers_no #TODO: Do you need this ?
            max = inlers_no
            e = outliers_no / len(matches_for_image)
            N = np.log(1 - p) / np.log(1 - np.power((1 - e), s))
            N = int(np.floor(N))
            no_iterations = N
            if(k > N or k >= MAX_RANSAC_ITERS): # this is saying if the max number of iterations you should have run is N, but you already did k > N then no point continuing
                return inlers_no, outliers_no, k, best_model, inliers

        if (k >= MAX_RANSAC_ITERS):
            return inlers_no, outliers_no, k, best_model, inliers
    return inlers_no, outliers_no, k, best_model, inliers

def run_ransac_modified(matches_for_image, distribution):
    s = 4  # or minimal_sample_size
    p = 0.99 # this is a typical value
    # number of iterations (http://www.cse.psu.edu/~rtc12/CSE486/lecture15.pdf and https://youtu.be/5E5n7fhLHEM?list=PLTBdjV_4f-EKeki5ps2WHqJqyQvxls4ha&t=428)
    # also reddit post https://www.reddit.com/r/computervision/comments/gikj1s/can_somebody_help_me_understand_ransac/
    no_iterations = MAX_RANSAC_ITERS  # can set this to whatever you want to start with
    k = 0
    distCoeffs = np.zeros((5, 1))  # assume zero for now
    threshold = 8.0 # same as opencv
    max = np.iinfo(np.int32).min
    best_model = {}

    while k < no_iterations:
        k = k + 1
        # pick 4 random matches (assume they are inliers)
        # p = distribution, From docs: The probabilities associated with each entry in a. If not given the sample assumes a uniform distribution over all entries in a
        random_matches = np.random.choice(len(matches_for_image), s , p = distribution, replace=False)

        # get 3D and 2D points
        obj_points = matches_for_image[(random_matches), 2:5]
        img_points = matches_for_image[(random_matches), 0:2]

        Rt = get_pose(img_points, obj_points)
        inliers, _ = model_fit(matches_for_image, Rt, threshold)

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
            if(k > N or k >= MAX_RANSAC_ITERS): # this is saying if the max number of iterations you should have run is N, but you already did k > N then no point continuing
                return inlers_no, outliers_no, k, best_model, inliers

        if (k >= MAX_RANSAC_ITERS):
            return inlers_no, outliers_no, k, best_model, inliers
    return inlers_no, outliers_no, k, best_model, inliers

# for findSupport()
# from emails - with author
# Yes, it is okay.
# A better one is considering also errors of points. For example:
# sum_errors = 0; num_inliers = 0
# for pt in points:
#    error = estimator.getError(model, pt) # e.g., estimator is PnP,
# error function computes reprojection error
#    if error < threshold:
#        sum_errors += error
#        num_inliers += 1
# sum_errors += (len(points) - num_inliers) * threshold
# So, you can compare models either by number of inliers (higher is
# better) or by sum of errors (lower is better).

def prosac(sorted_matches):
    # TODO: move this distCoeffs out!
    distCoeffs = np.zeros((5, 1))  # assume zero for now
    threshold = 8.0  # 8.0 same as opencv

    CORRESPONDENCES = sorted_matches.shape[0]
    isInlier = np.zeros([1,CORRESPONDENCES])
    SAMPLE_SIZE = 4
    MAX_OUTLIERS_PROPORTION = 0.8
    P_GOOD_SAMPLE = 0.99
    TEST_NB_OF_DRAWS = MAX_RANSAC_ITERS
    TEST_INLIERS_RATIO = 0.5
    BETA = 0.01
    ETA0 = 0.05

    def niter_RANSAC(p, epsilon, s, Nmax):
        if(Nmax == -1):
            Nmax = np.iinfo(np.int32).max
        if(not (Nmax >= 1)):
            raise RuntimeError("Nmax has to be positive")
        if(epsilon <= 0):
            return 1
        logarg = - np.exp(s * np.log(1 - epsilon))
        logval = np.log(1 + logarg)
        N = np.log(1 - p) / logval
        if(logval < 0 and N < Nmax):
            return np.ceil(N)
        return Nmax

    def Imin(m, n, beta):
        mu = n * beta
        sigma = np.sqrt(n * beta * (1 - beta))
        return  np.ceil(m + mu + sigma * np.sqrt(2.706))

    def findSupport(n, isInlier):
        #n is N and it is not used here
        total_inliers = isInlier.sum() # this can change to a another function (i.e kernel) as it is, it is too simple ?
        return total_inliers, isInlier

    N = CORRESPONDENCES
    m = SAMPLE_SIZE
    T_N = niter_RANSAC(P_GOOD_SAMPLE,MAX_OUTLIERS_PROPORTION,SAMPLE_SIZE,-1)
    beta = BETA
    I_N_min = (1 - MAX_OUTLIERS_PROPORTION)*N
    logeta0 = np.log(ETA0)

    n_star = N
    I_n_star = 0
    I_N_best = 0
    t = 0
    n = m
    T_n = T_N

    for i in range(m):
        T_n = T_n * (n - i) / (N - i)

    T_n_prime = 1
    k_n_star = T_N

    best_model = {}

    while (((I_N_best < I_N_min) or t <= k_n_star) and t < T_N and t <= TEST_NB_OF_DRAWS):
        model = {}
        inliers = []
        t = t + 1

        if ((t > T_n_prime) and (n < n_star)):
            T_nplus1 = (T_n * (n+1)) / (n+1-m)
            n = n + 1
            T_n_prime = T_n_prime + np.ceil(T_nplus1 - T_n)
            T_n = T_nplus1

        if (t > T_n_prime):
            pts_idx = np.random.choice(n, m, replace=False)
        else:
            pts_idx = np.append(np.random.choice(n-1, m-1, replace=False), n-1)

        sample = sorted_matches[pts_idx]

        # 3. Model parameter estimation
        obj_points = sample[:, 2:5]
        img_points = sample[:, 0:2]

        Rt = get_pose(img_points, obj_points)
        inliers, inliers_indices = model_fit(sorted_matches, Rt, threshold)
        isInlier = np.zeros([1, CORRESPONDENCES])
        isInlier[0, inliers_indices] = 1

        # 4. Model verification
        I_N, isInlier = findSupport(N, isInlier)

        if(I_N > I_N_best):
            I_N_best = I_N
            n_best = N
            I_n_best = I_N
            best_model = model

            if(1):
                epsilon_n_best = I_n_best / n_best
                I_n_test = I_N
                for n_test in range(N, m, -1):
                    if (not (n_test >= I_n_test)):
                        raise RuntimeError("Loop invariant broken: n_test >= I_n_test")
                    if ( (I_n_test * n_best > I_n_best * n_test) and (I_n_test > epsilon_n_best * n_test + np.sqrt(n_test * epsilon_n_best * (1 - epsilon_n_best) * 2.706) )):
                        if (I_n_test < Imin(m, n_test, beta)):
                            break
                        n_best = n_test
                        I_n_best = I_n_test
                        epsilon_n_best = I_n_best / n_best
                    I_n_test = I_n_test - isInlier[0, n_test - 1]

            if (I_n_best * n_star > I_n_star * n_best):
                if(not (n_best >= I_n_best)):
                    raise RuntimeError("Assertion not respected: n_best >= I_n_best")
                n_star = n_best
                I_n_star = I_n_best
                k_n_star = niter_RANSAC(1 - ETA0, 1 - I_n_star / n_star, m, T_N)
    
    # return values for readability
    inlier_no = I_N
    outliers_no = len(sorted_matches) - I_N
    iterations = t
    return inlier_no, outliers_no, iterations, best_model, inliers