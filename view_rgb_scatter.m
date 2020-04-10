% close all;
% 
% points_rgb = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/all_rgb_points3D.txt');
% 
% R = points_rgb(:,2);
% G = points_rgb(:,3);
% B = points_rgb(:,4);
% % s = 24;
% % 
% % figure(1)
% % for i = 1 : length(points_rgb)-75405
% %     scatter3(R(i),G(i),B(i), 'Marker', '.');
% %     hold on
% % end

gradients = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/gradients.txt');
