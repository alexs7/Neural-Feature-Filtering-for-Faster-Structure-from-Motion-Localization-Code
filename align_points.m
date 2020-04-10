close all;

colmapPoints = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/1/all_xyz_points3D.txt');
arcore_points = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/2/all_xyz_points3D.txt');

colmapPoints = colmapPoints(1:20000,2:4);
arcore_points = arcore_points(1:20000,2:4);

[d, Z, transform] = procrustes(arcore_points(1:20000,:), colmapPoints(1:20000,:));

figure
plot3(arcore_points(:,1),arcore_points(:,2),arcore_points(:,3),'r*');
hold on
plot3(Z(:,1),Z(:,2),Z(:,3),'b*');
hold on