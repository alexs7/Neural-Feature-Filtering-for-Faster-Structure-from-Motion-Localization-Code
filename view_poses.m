clear all;
close all;

points3D_start = importdata('points3D_start.txt');
points3D_start = points3D_start(:,1:3)

figure
pcshow(points3D_start,'MarkerSize', 4);
hold on

dinfo = dir('colmap_poses/*.txt');
for i = 1 : length(dinfo)
    pose = importdata(fullfile('colmap_poses/', dinfo(i).name));
    rotm = pose(1:3,1:3);
    tvec = pose(1:3,4);
    camera_location = -inv(rotm) * tvec;
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.03, 'Color', [1, 0, 0]);
end

hold on

dinfo = dir('final_poses/*.txt');
for i = 1 : length(dinfo)
    pose = importdata(fullfile('final_poses/', dinfo(i).name));
    rotm = pose(1:3,1:3);
    tvec = pose(1:3,4);
    camera_location = -inv(rotm) * tvec;
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.03, 'Color', [0, 1, 0]);
end
