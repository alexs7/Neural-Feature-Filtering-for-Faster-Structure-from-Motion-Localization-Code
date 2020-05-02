close all;
clear all;

% visibility_matrix_original = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/visibility_matrix_original.txt');
visibility_matrix_new = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/visibility_matrix_new.txt');
% sum_over_columns_new = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/sum_over_columns_new.txt');

visibility_matrix_new_sfm = visibility_matrix_new(1:130, :);


% figure
% plot(sum(visibility_matrix_new,1))
% % figure
% % plot(sum(visibility_matrix_original,1))
% 
x = 1:size(visibility_matrix_new,2);
y = 1:size(visibility_matrix_new,1);
[X,Y] = meshgrid(x,y);
Z = visibility_matrix_new(y,x);
surf(X,Y,Z);