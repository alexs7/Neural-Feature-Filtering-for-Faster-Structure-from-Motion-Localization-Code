% close all;
% 
% dist = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/1k/heatmap_matrix_avg_points_values_0.5.txt');
% figure
% bar(dist);
% 
% dist1 = importdata('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/visibility_matrices/1k/reliability_scores_0.5.txt');
% figure
% bar(dist1);

RT = [[ 0.84  0.26 -0.48 -2.68]
     [-0.1   0.94  0.32  1.08]
     [ 0.54 -0.22  0.82 -4.33]];

K_old = [[507.69,   0.  , 238.19],
           [  0.  , 507.62, 320.08],
           [  0.  ,   0.  ,   1.  ]];
       
K = [[545.971,   0.   , 238.19 ],
       [  0.   , 545.971, 320.08 ],
       [  0.   ,   0.   ,   1.   ]];
   
point3D = [12.46, -5.8,   5.73 1];

temp = RT * point3D';
point2D = K * temp(1:3);
point2D = point2D ./ point2D(3)
% point2D gt 469.8   68.31