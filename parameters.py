"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np

'''
List of shared parameters 
''' 
class Parameters(object):

    features_no = "1k"  # colmap_features_no can be "2k", "1k", "0.5k", "0.25k"
    exponential_decay_value = 0.5  # exponential_decay can be any of 0.1 to 0.9

    avg_descs_base_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/1k/avg_descs_base.npy"
    avg_descs_live_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/1k/avg_descs_live.npy"

    # RANSAC Comparison save locations
    matches_1_ransac_1_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_1_ransac_1_images_pose_" + str(
        exponential_decay_value) + ".npy"
    matches_1_ransac_1_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_1_ransac_1_data_" + str(
        exponential_decay_value) + ".npy"
    matches_1_prosac_1_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_1_prosac_1_images_pose_" + str(
        exponential_decay_value) + ".npy"
    matches_1_prosac_1_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_1_prosac_1_data_" + str(
        exponential_decay_value) + ".npy"

    matches_2_ransac_1_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_2_ransac_1_images_pose_" + str(
        exponential_decay_value) + ".npy"
    matches_2_ransac_1_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_2_ransac_1_data_" + str(
        exponential_decay_value) + ".npy"
    matches_2_ransac_2_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_2_ransac_2_images_pose_" + str(
        exponential_decay_value) + ".npy"
    matches_2_ransac_2_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_2_ransac_2_data_" + str(
        exponential_decay_value) + ".npy"
    matches_2_prosac_1_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_2_prosac_1_images_pose_" + str(
        exponential_decay_value) + ".npy"
    matches_2_prosac_1_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_2_prosac_1_data_" + str(
        exponential_decay_value) + ".npy"
    matches_2_prosac_2_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_2_prosac_2_images_pose_" + str(
        exponential_decay_value) + ".npy"
    matches_2_prosac_2_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_2_prosac_2_data_" + str(
        exponential_decay_value) + ".npy"
    matches_2_prosac_3_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_2_prosac_3_images_pose_" + str(
        exponential_decay_value) + ".npy"
    matches_2_prosac_3_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_2_prosac_3_data_" + str(
        exponential_decay_value) + ".npy"
    matches_2_prosac_4_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_2_prosac_4_images_pose_" + str(
        exponential_decay_value) + ".npy"
    matches_2_prosac_4_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_2_prosac_4_data_" + str(
        exponential_decay_value) + ".npy"
    matches_2_prosac_5_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_2_prosac_5_images_pose_" + str(
        exponential_decay_value) + ".npy"
    matches_2_prosac_5_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_2_prosac_5_data_" + str(
        exponential_decay_value) + ".npy"

    matches_3_ransac_1_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_3_ransac_1_images_pose_" + str(
        exponential_decay_value) + ".npy"
    matches_3_ransac_1_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_3_ransac_1_data_" + str(
        exponential_decay_value) + ".npy"
    matches_3_ransac_2_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_3_ransac_2_images_pose_" + str(
        exponential_decay_value) + ".npy"
    matches_3_ransac_2_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_3_ransac_2_data_" + str(
        exponential_decay_value) + ".npy"
    matches_3_prosac_1_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_3_prosac_1_images_pose_" + str(
        exponential_decay_value) + ".npy"
    matches_3_prosac_1_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_3_prosac_1_data_" + str(
        exponential_decay_value) + ".npy"
    matches_3_prosac_2_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_3_prosac_2_images_pose_" + str(
        exponential_decay_value) + ".npy"
    matches_3_prosac_2_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_3_prosac_2_data_" + str(
        exponential_decay_value) + ".npy"
    matches_3_prosac_3_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_3_prosac_3_images_pose_" + str(
        exponential_decay_value) + ".npy"
    matches_3_prosac_3_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_3_prosac_3_data_" + str(
        exponential_decay_value) + ".npy"
    matches_3_prosac_4_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_3_prosac_4_images_pose_" + str(
        exponential_decay_value) + ".npy"
    matches_3_prosac_4_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_3_prosac_4_data_" + str(
        exponential_decay_value) + ".npy"
    matches_3_prosac_5_path_poses = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_3_prosac_5_images_pose_" + str(
        exponential_decay_value) + ".npy"
    matches_3_prosac_5_path_data = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/RANSAC_results/" + features_no + "/matches_3_prosac_5_data_" + str(
        exponential_decay_value) + ".npy"

    # PROSAC sorting values for matches indices
    use_ransac_dist = -1
    lowes_distance_inverse_ratio_index = 0
    heatmap_val_index = 2
    reliability_score_ratio_index = 3
    custom_score_index = 4
    higher_neighbour_score_index = 5
    reliability_score_index = 6

    # 29/06/2020 - My addition
    live_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/all_models/live_model/images.bin"
    base_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/all_models/base_model/images.bin"
    gt_model_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/all_models/ground_truth_model/images.bin"
    live_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/all_models/live_model/points3D.bin"
    base_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/all_models/base_model/points3D.bin"
    gt_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/all_models/ground_truth_model/points3D.bin"

    live_db_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/all_models/live_model/database.db"
    base_db_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/all_models/base_model/database.db"
    qt_db_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/all_models/ground_truth_model/database.db"

    query_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/all_models/ground_truth_model/query_name.txt"
    query_images_camera_intrinsics = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/all_models/ground_truth_model/intrinsics_query_images_camera.txt"
    query_images_camera_intrinsics_old = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/matrices/pixel_intrinsics_low_640_portrait.txt"

    matches_base_save_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/" + features_no + "/matches_base.npy"
    matches_live_save_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/feature_matching/" + features_no + "/matches_live.npy"

    # Parameters.no_images_per_session: Number of images per session. This is hardcoded for now, but since images are sorted by name, i.e by time in the database,
    # then you can use these numbers to get images from each session. The numbers need to be sorted by session though. First is number of base model images.
    no_images_per_session = [210, 85, 87, 79, 86, 83, 85, 90, 86, 84, 79, 95]
    ratio_test_val = 0.9

    # FLANN parameters for float descriptors
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=32)  # or pass empty dictionary

    kFeatureMatchRatioTest = 0.7
