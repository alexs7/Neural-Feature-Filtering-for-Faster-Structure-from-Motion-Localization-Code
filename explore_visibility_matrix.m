close all;
clear all;

visibility_matrix = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/visibility_matrix_new.txt');

% figure
% plot(sum(visibility_matrix_new,1))
% % figure
% % plot(sum(visibility_matrix_original,1))
% 
x = 1:size(visibility_matrix,2);
y = 1:size(visibility_matrix,1);
[X,Y] = meshgrid(x,y);
Z = visibility_matrix(y,x);
surf(X,Y,Z);