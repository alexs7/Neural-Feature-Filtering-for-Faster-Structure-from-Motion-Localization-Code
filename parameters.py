import os

class Parameters(object):

    def __init__(self, base_path):
        self.benchmarks_iters = 15
        self.avg_descs_base_path = os.path.join(base_path , "avg_descs_base.npy")
        self.avg_descs_live_path = os.path.join(base_path ,"avg_descs_live.npy")

        self.per_image_decay_matrix_path = os.path.join(base_path , "heatmap_matrix_avg_points_values.npy")
        self.per_session_decay_matrix_path = os.path.join(base_path , "reliability_scores.npy")
        self.binary_visibility_matrix_path = os.path.join(base_path , "binary_visibility_values.npy")

        self.matches_base_save_path = os.path.join(base_path , "matches_base.npy")
        self.matches_live_save_path = os.path.join(base_path , "matches_live.npy")

        # 29/06/2020 - My addition
        self.live_model_images_path = os.path.join(base_path , "live/model/images.bin")
        self.base_model_images_path = os.path.join(base_path , "base/model/0/images.bin")
        self.gt_model_images_path = os.path.join(base_path , "gt/model/images.bin")

        self.live_model_points3D_path = os.path.join(base_path , "live/model/points3D.bin")
        self.base_model_points3D_path = os.path.join(base_path , "base/model/0/points3D.bin")
        self.gt_model_points3D_path = os.path.join(base_path , "gt/model/points3D.bin")

        self.live_db_path = os.path.join(base_path , "live/database.db")
        self.base_db_path = os.path.join(base_path , "base/database.db")
        self.gt_db_path = os.path.join(base_path , "gt/database.db")

        # not the session ones!
        self.query_images_path = os.path.join(base_path , "gt/query_name.txt")

        self.gt_model_cameras_path = os.path.join(base_path , "gt/model/cameras.bin")

        self.save_results_path = os.path.join(base_path , "results.npy")

        # Parameters.no_images_per_session: Number of images per session. The numbers need to be sorted by session though. First is number of base model images.
        self.no_images_per_session_path = os.path.join(base_path , "live/session_lengths.txt")

        self.ratio_test_val = 0.9

        # This is the scale you will have to multiply your COLMAP model's acquired camera centers distance with.
        # Pass this in pose evaluator, and it will be multiplied with the distance of the gt camera center and your estimated camera center
        # from your COLMAP model. This is valid for ARCORE only
        self.ARCORE_scale_path = os.path.join(base_path , "scale.txt")

        self.ml_models_trained = [
            "Classifier w/ top 10% matches" ,
            "Classifier using all matches"  ,
            "Classifier and Regressor w/ image score" ,
            "Classifier and Regressor w/ score per session" ,
            "Classifier and Regressor w/ visibility score" ,
            "Regressor w/ score per image" ,
            "Regressor w/ score per session" ,
            "Regressor w/ visibility score" ,
            "Combined w/ score per image" ,
            "Combined w/ score per session" ,
            "Combined w/ visibility score"
        ]

        self.baseline_methods = {
            "Rndm 10%": "ML_data/prepared_data/random_output",
            "All (~800)": "ML_data/prepared_data/baseline_output"
        }

        self.ml_methods = { # my models, full names
            "Class, top mtchs": "Classifier w/ top 10% matches",  # 0
            "Class, all mtchs": "Classifier using all matches",  # 1
            "C & R, s.p.i": "Classifier and Regressor w/ image score",  # 2
            "C & R, s.p.s": "Classifier and Regressor w/ score per session",  # 3
            "C & R, s.p.v": "Classifier and Regressor w/ visibility score",  # 4
            "R, s.p.i": "Regressor w/ score per image",  # 5
            "R, s.p.s": "Regressor w/ score per session",  # 6
            "R, s.p.v": "Regressor w/ visibility score",  # 7
            "CB, s.p.i": "Combined w/ score per image",  # 8
            "CB, s.p.s": "Combined w/ score per session",  # 9
            "CB, s.p.v": "Combined w/ visibility score",  # 10
            "Rd C & R s.p.i": "Class. and Regr. w/ score per image, dist. RANSAC",  # 11
            "Rd C & R s.p.s": "Class. and Regr. w/ score per session, dist. RANSAC",  # 12
            "Rd C & R s.p.v": "Class. and Regr. w/ visibility score, dist. RANSAC",  # 13
            "PRSC R, s.p.i": "Regressor w/ score per image, PROSAC",  # 14
            "PRSC R, s.p.s": "Regressor w/ score per session, PROSAC",  # 15
            "PRSC R, s.p.v": "Regressor w/ visibility score, PROSAC",  # 16
            "PRSC CB, s.p.i": "Combined w/ score per image, PROSAC",  # 17
            "PRSC CB, s.p.s": "Combined w/ score per session, PROSAC",  # 18
            "PRSC CB, s.p.v": "Combined w/ visibility score, PROSAC"  # 19
        }

        self.ml_methods_matches_map = {  # the methods and the matches index they used
            "Class, top mtchs": 0,
            "Class, all mtchs": 1,
            "C & R, s.p.i": 2,
            "C & R, s.p.s": 3,
            "C & R, s.p.v": 4,
            "R, s.p.i": 5,
            "R, s.p.s": 6,
            "R, s.p.v": 7,
            "CB, s.p.i": 8,
            "CB, s.p.s": 9,
            "CB, s.p.v": 10,
            "Rd C & R s.p.i": 2,
            "Rd C & R s.p.s": 3,
            "Rd C & R s.p.v": 4,
            "PRSC R, s.p.i": 5,
            "PRSC R, s.p.s": 6,
            "PRSC R, s.p.v": 7,
            "PRSC CB, s.p.i": 8,
            "PRSC CB, s.p.s": 9,
            "PRSC CB, s.p.v": 10
        }

        self.comparison_methods = {
            "Pred. M. 2014": "predicting_matchability_comparison_data",
            "Pred. M. 2014 VRF": "ml_models_vanillia_comparison_data",
            "M. or no M. 2020": "match_or_no_match_comparison_data"
        }

        self.consensus_types = ['ransac_base', 'prosac_base', 'ransac_live', 'ransac_dist_heatmap_val',
                                'ransac_dist_visibility_score', 'inverse_lowes_ratio', 'reliability_higher_neighbour_heatmap_value',
                                'reliability_higher_neighbour_score', 'higher_neighbour_visibility_score', 'lowes_by_reliability_score_ratio']