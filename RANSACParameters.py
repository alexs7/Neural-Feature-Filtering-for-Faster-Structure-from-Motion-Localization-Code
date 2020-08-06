class RANSACParameters(object):
    # PROSAC sorting values for matches indices
    use_ransac_dist = -1
    lowes_distance_inverse_ratio_index = 0
    heatmap_val_index = 2
    reliability_score_ratio_index = 3
    custom_score_index = 4
    higher_neighbour_score_index = 5
    reliability_score_index = 6
    heatmap_val_ratio_index = 7
    higher_neighbour_val_index = 8
    custom_score_index_2 = 9

    prosac_value_titles = {lowes_distance_inverse_ratio_index: "Inverse Lowe's Ratio",
                           heatmap_val_index: "Heatmap Value",
                           reliability_score_index: "Reliability Score",
                           reliability_score_ratio_index: "Reliability Score Ratio",
                           custom_score_index: "Lowes by reliability score ratio",
                           custom_score_index_2: "Lowes by heatmap value ratio",
                           higher_neighbour_score_index: "Reliability Higher Neighbour Score",
                           heatmap_val_ratio_index: "Heatmap Value Ratio",
                           higher_neighbour_val_index: "Reliability Higher Neighbour Heatmap Value"}

    ransac_prosac_iterations = 1000  # I used 1000 for dev
    ransac_prosac_error_threshold = 5.0  # 8.0 default, 5 from the Benchamrking 6DoF long term localization