close all;

colmapPoints = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/colmap_points.txt');
arcore_points = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/arcore_points.txt');

[d, Z, transform] = procrustes(arcore_points(1:250,:), colmapPoints(1:250,:));

figure
plot3(arcore_points(:,1),arcore_points(:,2),arcore_points(:,3),'g*');
hold on
plot3(Z(:,1),Z(:,2),Z(:,3),'b*');
hold on