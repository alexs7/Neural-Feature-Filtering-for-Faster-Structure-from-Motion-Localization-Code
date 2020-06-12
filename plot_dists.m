close all;

dist = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/modified_results_t_1k_0.5.txt');
figure
bar(dist);

dist = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/pose_evaluator/vanillia_results_t_1k_0.5.txt');
figure
bar(dist);