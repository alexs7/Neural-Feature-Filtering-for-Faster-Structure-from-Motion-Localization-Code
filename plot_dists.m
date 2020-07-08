close all;

dist = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/1k/heatmap_matrix_avg_points_values_0.5.txt');
figure
bar(dist);

dist1 = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/1k/reliability_scores_0.5.txt');
figure
bar(dist1);

