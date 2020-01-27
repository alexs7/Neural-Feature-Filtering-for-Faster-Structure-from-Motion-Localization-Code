clear all;
close all;

points3D_start = importdata('points3D_start.txt');
points3D_start = points3D_start(:,1:3);

figure
pcshow(points3D_start,'MarkerSize', 8);
hold on
% plot3(0,0,0,'g*');

% set(gca,'color','w');
xlabel('X');
ylabel('Y');
zlabel('Z');
hold on

minus_cameras = 0;

gt_cam_loc = [];
dinfo = dir('colmap_poses/*.txt');
for i = 1 : length(dinfo) - minus_cameras
    pose = importdata(fullfile('colmap_poses/', dinfo(i).name));
    rotm = pose(1:3,1:3);
    tvec = pose(1:3,4);
    
    camera_location = -inv(rotm) * tvec;
    
    gt_cam_loc = [gt_cam_loc ; camera_location'];
    
    % set to origin
%     if(i == 1)
%         diff = camera_location;
%         camera_location = [0 0 0];
%     else
%         camera_location = camera_location - diff;
%     end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.03, 'Color', [1, 0, 0]);
end

hold on

fp_cam_loc = [];
dinfo = dir('final_poses/*.txt');
for i = 1 : length(dinfo) - minus_cameras
    pose = importdata(fullfile('final_poses/', dinfo(i).name));
    rotm =  pose(1:3,1:3);
    tvec =  pose(1:3,4);
    
    camera_location = -inv(rotm) * tvec;
    
    fp_cam_loc = [fp_cam_loc ; camera_location'];

    % set to origin
%     if(i == 1)
%         diff = camera_location;
%         camera_location = [0 0 0];
%     else
%         camera_location = camera_location - diff;
%     end
    
%     plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.03, 'Color', [0, 1, 0], 'Opacity', 0.1);
end

gt_cam_loc = load('gt_cam_loc.mat');
fp_cam_loc = load('fp_cam_loc.mat');
[d,Z,transform] = procrustes(gt_cam_loc, fp_cam_loc);
c = transform.c;
T = transform.T;
b = transform.b;

hold on;
%plot3(Z(:,1), Z(:,2), Z(:,3),'y*');

% draw final poses
dinfo = dir('final_poses/*.txt');
for i = 1 : length(dinfo) - minus_cameras
    pose = importdata(fullfile('final_poses/', dinfo(i).name));
    rotm =  pose(1:3,1:3);
    tvec =  pose(1:3,4);
    
    camera_location = -inv(rotm) * tvec;
    
    camera_location = b * camera_location' * T + c(1,:);
    
    % save new pose
    t_new = -rotm * camera_location';
    pose_new = [rotm t_new ; 0 0 0 1];
    save(strcat(strcat('matlab_res/', dinfo(i).name ), '.mat'),'pose_new');
    
    % set to origin
%     if(i == 1)
%         diff = camera_location;
%         camera_location = [0 0 0];
%     else
%         camera_location = camera_location - diff;
%     end
    
    plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.03, 'Color', [0, 1, 0], 'Opacity', 0.1);
    
    
end

% hold on
% 
% dinfo = dir('cameraPose_dp_oriented/*.txt');
% for i = 1 : length(dinfo) - minus_cameras
%     pose = importdata(fullfile('cameraPose_dp_oriented/', dinfo(i).name));
%     rotm =  pose(1:3,1:3);
%     tvec =  pose(1:3,4);
%     
%     camera_location = -inv(rotm) * tvec;
%     
%     % set to origin
%     if(i == 1)
%         diff = camera_location;
%         camera_location = [0 0 0];
%     else
%         camera_location = camera_location - diff;
%     end
%     
%     plotCamera('Location', camera_location, 'Orientation', rotm, 'Size', 0.03, 'Color', [0, 0, 1], 'Opacity', 0.1);
% end
