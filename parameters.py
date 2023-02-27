import os

class Parameters(object):

    def __init__(self, base_path):
        self.results_path = os.path.join(base_path,"results")
        self.degenerate_poses_path = os.path.join(self.results_path, "degenerate_poses.npy")
        self.aggregated_results_csv = os.path.join(self.results_path,"evaluation_results_2022_aggregated.csv")
        self.cmu_only_all_slices_aggregated_results_csv = os.path.join(self.results_path,"evaluation_results_2022_aggregated_all_slices.csv")
        self.avg_descs_base_path = os.path.join(base_path,"avg_descs_base.npy")
        self.avg_descs_live_path = os.path.join(base_path,"avg_descs_live.npy")

        # This is for MnM (Match and No Match) and PM (Predicting Matchability)
        self.match_or_no_match_comparison_data = "models_for_match_no_match"  # also contains models
        self.mnm_path = os.path.join(base_path, self.match_or_no_match_comparison_data)
        self.gt_db_path_mnm = os.path.join(self.mnm_path, "gt/database.db")
        self.avg_descs_gt_path_mnm = os.path.join(self.mnm_path, "gt", "avg_descs_gt_mnm.npy")
        self.gt_model_points3D_path_mnm = os.path.join(self.mnm_path, "gt/output_opencv_sift_model/points3D.bin")
        self.gt_model_images_path_mnm = os.path.join(self.mnm_path, "gt/output_opencv_sift_model/images.bin")
        self.gt_model_cameras_path_mnm = os.path.join(self.mnm_path, "gt/output_opencv_sift_model/cameras.bin")
        self.query_gt_images_txt_path_mnm = os.path.join(self.mnm_path, "gt/query_name.txt")
        self.mnm_trained_model_path_mnm = os.path.join(self.mnm_path, "trained_model_pairs_no_8000.xml")
        self.predicting_matchability_comparison_data = "predicting_matchability_comparison_data"

        # 30/12/2022 still using the old names for filenames (_matrix) - nvm
        self.per_image_decay_scores_path = os.path.join(base_path, "per_image_score.npy")
        self.per_session_decay_scores_path = os.path.join(base_path, "per_session_score.npy")
        self.binary_visibility_scores_path = os.path.join(base_path, "visibility_scores.npy")
        self.live_points_3D_ids_file_path = os.path.join(base_path,"live_points_3D_ids.npy")
        self.base_points_3D_ids_file_path = os.path.join(base_path,"base_points_3D_ids.npy")

        self.matches_base_save_path = os.path.join(base_path,"matches_base.npy")
        self.matches_live_save_path = os.path.join(base_path,"matches_live.npy")

        self.points3D_seen_per_image_base = os.path.join(base_path, "points3D_seen_per_image_base.npy")
        self.points3D_seen_per_image_live = os.path.join(base_path, "points3D_seen_per_image_live.npy")

        # 29/06/2020 - My addition
        # 08/12/2022 - changed from "base/model/0/points3D.bin" to "base/model/points3D.bin"
        # same for images.bin, as the folder /0/ is not created anymore
        self.live_model_images_path = os.path.join(base_path,"live/model/images.bin")
        self.base_model_images_path = os.path.join(base_path,"base/model/images.bin")
        self.gt_model_images_path = os.path.join(base_path,"gt/model/images.bin")

        self.live_model_points3D_path = os.path.join(base_path,"live/model/points3D.bin")
        self.base_model_points3D_path = os.path.join(base_path,"base/model/points3D.bin")
        self.gt_model_points3D_path = os.path.join(base_path,"gt/model/points3D.bin")

        self.live_db_path = os.path.join(base_path,"live/database.db")
        self.base_db_path = os.path.join(base_path,"base/database.db")
        self.gt_db_path = os.path.join(base_path,"gt/database.db")

        # not the session ones!
        self.query_images_path = os.path.join(base_path,"gt/query_name.txt")

        self.gt_model_cameras_path = os.path.join(base_path,"gt/model/cameras.bin")

        self.save_results_path = os.path.join(base_path,"results.npy")
        self.save_results_csv_path = os.path.join(base_path,"results.csv")

        # Parameters.no_images_per_session: Number of images per session. The numbers need to be sorted by session though. First is number of base model images.
        # The gt session_lengths is not used.
        self.no_images_per_session_path = os.path.join(base_path,"live/session_lengths.txt")

        self.ratio_test_val = 0.9

        # This is the scale you will have to multiply your COLMAP model's acquired camera centers distance with.
        # Pass this in pose evaluator, and it will be multiplied with the distance of the gt camera center and your estimated camera center
        # from your COLMAP model. This is valid for ARCORE only
        # 30/12/2022 - New path added
        self.ARCORE_scale_path = os.path.join(base_path, "ML_data", "prepared_data", "scale.npy")

        self.debug_images_path = os.path.join(base_path, "debug_images")
        self.debug_images_base_path = os.path.join(base_path, "debug_images_base")
        self.debug_images_live_path = os.path.join(base_path, "debug_images_live")
        self.debug_images_gt_path = os.path.join(base_path, "debug_images_gt")