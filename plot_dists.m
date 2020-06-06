close all;

dist = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/vanilla_ransac_results_t_1k_0.5_weighted.txt');
figure
bar(dist);

dist = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/modified_ransac_results_t_1k_0.5_weighted.txt');
figure
bar(dist);