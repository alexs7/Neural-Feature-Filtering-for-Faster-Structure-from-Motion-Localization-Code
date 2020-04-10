moving = pcread('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/1/model.ply');
fixed = pcread('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/2/model.ply');

resultT = pcregistericp(moving,fixed, 'Extrapolate', true);
resultT.T
rotm = resultT.T(1:3,1:3)';
quat = rotm2quat(rotm)

ptCloudTformed = pctransform(moving,resultT);
pcwrite(ptCloudTformed,'/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/result','PLYFormat','binary');



