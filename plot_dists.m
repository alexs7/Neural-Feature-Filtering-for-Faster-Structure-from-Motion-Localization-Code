close all;

dist = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/1k/heatmap_matrix_avg_points_values_0.5.txt');
figure
points_static_indices = dist > mean(dist);

bar(dist(points_static_indices));

dist = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/rgb_avg.txt');
figure
bar(dist(points_static_indices));