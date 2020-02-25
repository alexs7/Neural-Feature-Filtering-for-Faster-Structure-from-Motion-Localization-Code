clear all;
close all;

% points3D_start = importdata('points3D_start.txt');
% points3D_start = points3D_start(:,1:3);

figure
pcshow([0,0,0],'MarkerSize', 8);
hold on
% plot3(0,0,0,'g*');

% set(gca,'color','w');
xlabel('X');
ylabel('Y');
zlabel('Z');
hold on

minus_cameras = 0;
step_size = 1;

ar_core_poses_locs = [];
arcore_relative_poses_locs = [];
global_poses_locs = [];
relative_poses_locs = [];
ar_core_camera_poses_locs = [];

dinfo = dir('poses_for_debug/ar_core_oriented_poses/*.txt');
for i = 1 : length(dinfo) - minus_cameras
    pose = importdata(fullfile('poses_for_debug/ar_core_oriented_poses/', dinfo(i).name));
    rotm =  pose(1:3,1:3);
    tvec =  pose(1:3,4);
    
    camera_location = -inv(rotm) * tvec;
    ar_core_poses_locs = [ar_core_poses_locs ; camera_location'];
    
    % set to origin
    if(i == 1)
        diff = camera_location;
        camera_location = [0 0 0];
    else
        camera_location = camera_location - diff;
    end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.05, 'Color', [1, 1, 0]);
end

hold on;

dinfo = dir('poses_for_debug/arcore_relative_poses/*.txt');
for i = 1 : length(dinfo) - minus_cameras
    pose = importdata(fullfile('poses_for_debug/arcore_relative_poses/', dinfo(i).name));
    rotm =  pose(1:3,1:3);
    tvec =  pose(1:3,4);
    
    camera_location = -inv(rotm) * tvec;
    arcore_relative_poses_locs = [ arcore_relative_poses_locs ; camera_location' ];
    
    % set to origin
    if(i == 1)
        diff = camera_location;
        camera_location = [0 0 0];
    else
        camera_location = camera_location - diff;
    end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.05, 'Color', [0, 1, 0]);
end

hold on;

dinfo = dir('poses_for_debug/ar_core_camera_poses/*.txt');
for i = 1 : length(dinfo) - minus_cameras
    pose = importdata(fullfile('poses_for_debug/ar_core_camera_poses/', dinfo(i).name));
    rotm =  pose(1:3,1:3);
    tvec =  pose(1:3,4);
    
    camera_location = -inv(rotm) * tvec;
    ar_core_camera_poses_locs = [ ar_core_camera_poses_locs ; camera_location' ];
    
    % set to origin
    if(i == 1)
        diff = camera_location;
        camera_location = [0 0 0];
    else
        camera_location = camera_location - diff;
    end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.05, 'Color', [1, 1, 1]);
end

hold on;

dinfo = dir('poses_for_debug/colmap_poses/*.txt');
for i = 1 : step_size :  length(dinfo) - minus_cameras
    pose = importdata(fullfile('poses_for_debug/colmap_poses/', dinfo(i).name));
    rotm = pose(1:3,1:3);
    tvec = pose(1:3,4);
    
    camera_location = -inv(rotm) * tvec;
    global_poses_locs = [ global_poses_locs ; camera_location' ];
    
    %set to origin
    if(i == 1)
        diff = camera_location;
        camera_location = [0 0 0];
    else
        camera_location = camera_location - diff;
    end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.05, 'Color', [1, 0, 0]);
end

hold on;

dinfo = dir('poses_for_debug/ar_core_and_colmap_relative_poses/*.txt');
for i = 1 : step_size :  length(dinfo) - minus_cameras
    pose = importdata(fullfile('poses_for_debug/ar_core_and_colmap_relative_poses/', dinfo(i).name));
    rotm = pose(1:3,1:3);
    tvec = pose(1:3,4);
    
    camera_location = -inv(rotm) * tvec;
    relative_poses_locs = [ relative_poses_locs ; camera_location' ];
    
    % set to origin
    if(i == 1)
        diff = camera_location;
        camera_location = [0 0 0];
    else
        camera_location = camera_location - diff;
    end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.05, 'Color', [0, 0, 1]);
end

% Procrustes analysis results

% between colmap poses and displayOrientedPoses
[d, Z, transform] = procrustes(global_poses_locs, ar_core_poses_locs);
figure
pcshow([0,0,0],'MarkerSize', 8);
hold on
% plot3(0,0,0,'g*');

% set(gca,'color','w');
xlabel('X');
ylabel('Y');
zlabel('Z');
hold on;

dinfo = dir('poses_for_debug/colmap_poses/*.txt');
for i = 1 : step_size :  length(dinfo) - minus_cameras
    pose = importdata(fullfile('poses_for_debug/colmap_poses/', dinfo(i).name));
    rotm = pose(1:3,1:3);
    tvec = pose(1:3,4);
    
    camera_location = -inv(rotm) * tvec;
    
    %set to origin
    if(i == 1)
        diff = camera_location;
        camera_location = [0 0 0];
    else
        camera_location = camera_location - diff;
    end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.05, 'Color', [1, 0, 0]);
end

hold on;

dinfo = dir('poses_for_debug/ar_core_oriented_poses/*.txt');
for i = 1 : length(dinfo) - minus_cameras
    pose = importdata(fullfile('poses_for_debug/ar_core_oriented_poses/', dinfo(i).name));
    rotm =  pose(1:3,1:3);
    tvec =  pose(1:3,4);
    
    camera_location = Z(i,:);
    
    % set to origin
    if(i == 1)
        diff = camera_location;
        camera_location = [0 0 0];
    else
        camera_location = camera_location - diff;
    end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.05, 'Color', [1, 1, 0]);
end

hold on;

% between colmap poses and ARCore relative poses

[d, Z, transform] = procrustes(global_poses_locs, arcore_relative_poses_locs);
figure
pcshow([0,0,0],'MarkerSize', 8);
hold on
% plot3(0,0,0,'g*');

% set(gca,'color','w');
xlabel('X');
ylabel('Y');
zlabel('Z');
hold on;

dinfo = dir('poses_for_debug/colmap_poses/*.txt');
for i = 1 : step_size :  length(dinfo) - minus_cameras
    pose = importdata(fullfile('poses_for_debug/colmap_poses/', dinfo(i).name));
    rotm = pose(1:3,1:3);
    tvec = pose(1:3,4);
    
    camera_location = -inv(rotm) * tvec;
    
    %set to origin
    if(i == 1)
        diff = camera_location;
        camera_location = [0 0 0];
    else
        camera_location = camera_location - diff;
    end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.05, 'Color', [1, 0, 0]);
end

hold on;

dinfo = dir('poses_for_debug/arcore_relative_poses/*.txt');
for i = 1 : length(dinfo) - minus_cameras
    pose = importdata(fullfile('poses_for_debug/arcore_relative_poses/', dinfo(i).name));
    rotm =  pose(1:3,1:3);
    tvec =  pose(1:3,4);
    
    camera_location = Z(i,:);
    
    % set to origin
    if(i == 1)
        diff = camera_location;
        camera_location = [0 0 0];
    else
        camera_location = camera_location - diff;
    end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.05, 'Color', [0, 1, 0]);
end

hold on;

% between colmap poses and ARCore camera plain poses

[d, Z, transform] = procrustes(global_poses_locs, ar_core_camera_poses_locs);
figure
pcshow([0,0,0],'MarkerSize', 8);
hold on
% plot3(0,0,0,'g*');

% set(gca,'color','w');
xlabel('X');
ylabel('Y');
zlabel('Z');
hold on;

dinfo = dir('poses_for_debug/colmap_poses/*.txt');
for i = 1 : step_size :  length(dinfo) - minus_cameras
    pose = importdata(fullfile('poses_for_debug/colmap_poses/', dinfo(i).name));
    rotm = pose(1:3,1:3);
    tvec = pose(1:3,4);
    
    camera_location = -inv(rotm) * tvec;
    
    %set to origin
    if(i == 1)
        diff = camera_location;
        camera_location = [0 0 0];
    else
        camera_location = camera_location - diff;
    end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.05, 'Color', [1, 0, 0]);
end

hold on;

dinfo = dir('poses_for_debug/ar_core_camera_poses/*.txt');
for i = 1 : length(dinfo) - minus_cameras
    pose = importdata(fullfile('poses_for_debug/ar_core_camera_poses/', dinfo(i).name));
    rotm =  pose(1:3,1:3);
    tvec =  pose(1:3,4);
    
    camera_location = Z(i,:);
    
    % set to origin
    if(i == 1)
        diff = camera_location;
        camera_location = [0 0 0];
    else
        camera_location = camera_location - diff;
    end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.05, 'Color', [1, 1, 1]);
end

hold on;

% between colmap poses and ARCore/COLMAP relative poses

[d, Z, transform] = procrustes(global_poses_locs, relative_poses_locs);
figure
pcshow([0,0,0],'MarkerSize', 8);
hold on
% plot3(0,0,0,'g*');

% set(gca,'color','w');
xlabel('X');
ylabel('Y');
zlabel('Z');
hold on;

dinfo = dir('poses_for_debug/colmap_poses/*.txt');
for i = 1 : step_size :  length(dinfo) - minus_cameras
    pose = importdata(fullfile('poses_for_debug/colmap_poses/', dinfo(i).name));
    rotm = pose(1:3,1:3);
    tvec = pose(1:3,4);
    
    camera_location = -inv(rotm) * tvec;
    
    %set to origin
    if(i == 1)
        diff = camera_location;
        camera_location = [0 0 0];
    else
        camera_location = camera_location - diff;
    end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.05, 'Color', [1, 0, 0]);
end

hold on;

dinfo = dir('poses_for_debug/ar_core_and_colmap_relative_poses/*.txt');
for i = 1 : length(dinfo) - minus_cameras
    pose = importdata(fullfile('poses_for_debug/ar_core_and_colmap_relative_poses/', dinfo(i).name));
    rotm =  pose(1:3,1:3);
    tvec =  pose(1:3,4);
    
    camera_location = Z(i,:);
    
    % set to origin
    if(i == 1)
        diff = camera_location;
        camera_location = [0 0 0];
    else
        camera_location = camera_location - diff;
    end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.05, 'Color', [0, 0, 1]);
end

hold on;

