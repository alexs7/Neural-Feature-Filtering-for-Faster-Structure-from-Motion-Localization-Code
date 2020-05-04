const fs = require('fs');
const express = require('express');
const bodyParser = require('body-parser');
const { execSync } = require('child_process');
const {getCurrentWindow, globalShortcut} = require('electron').remote;

//3D Objects
var phone_cam;
var anchor;
var colmap_points;
var arcore_points;
var scene;
var server;
var cameraDisplayOrientedPose;
var camera; //ThreeJS Camera
var controls;
var origin = new THREE.Vector3( 0, 0, 0 );
var red = 0xff0000;
var green = 0x00ff00;
var blue = 0x0000ff;
var yellow = 0xffff00;
var white = 0xffffff;
var orange = 0xffa500;
var pink = 0xFFC0CB;
var useCameraDisplayOrientedPose = true;
var camera_pose;
var local_camera_axes_points;
var x_axis_point;
var y_axis_point;
var z_axis_point;
var cameraWorldCenter;
var cameraWorldCenterPoint;
var debugAnchorPosition;
var debugAnchor;
var arCoreViewMatrix;
var arCoreProjMatrix;
var cameraPoseStringMatrix;
var totalPointsSum = 0;
var pointsSize = 0.08;

window.onload = function() {

    $(".one").click(function(){
        renderModel(1, red);
    });

    $(".two").click(function(){
        renderModel(2, blue);
    });

    $(".three").click(function(){
        renderModel(3, yellow);
    });

    $(".four").click(function(){
        renderModel(4, green);
    });

    $(".five").click(function(){
        renderModel(5, yellow);
    });

    $(".six").click(function(){
        renderModel(6, pink);
    });

    $(".seven").click(function(){
        renderModel(7, orange);
    });

    $(".eight").click(function(){
        renderModel(8, white);
    });

    $(".reset").click(function(){
        totalPointsSum = 0;
        clearScene();
    });

    $(".loadCompleteModel").click(function(){

        clearScene();

        const file_path = '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/all_xyz_points3D.txt';

        var data = fs.readFileSync(file_path);
        data = data.toString().split('\n');
        data.pop() // remove the last element ""

        console.log("loadCompleteModel " + data.length);

        var geometry = new THREE.Geometry();

        for (var i = 0; i < data.length; i++) {
            var line = data[i].split(' ');
            var x = parseFloat(line[1]);
            var y = parseFloat(line[2]);
            var z = parseFloat(line[3]);
            geometry.vertices.push(
                new THREE.Vector3(x, y, z)
            );
        }

        var material =  new THREE.PointsMaterial( { color: red, size: pointsSize } );
        var points = new THREE.Points( geometry, material );
        // points.scale.set(0.1,0.1,0.1);
        scene.add(points);
    });

    $(".loadCompressedModel_VMM").click(function(){

        clearScene();

        const file_path = '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/col_sum_points_camera_and_sum_mean.txt';

        var data = fs.readFileSync(file_path);
        data = data.toString().split('\n');
        data.pop() // remove the last element ""

        console.log("loadCompressedModel_VMM " + data.length);

        var geometry = new THREE.Geometry();

        for (var i = 0; i < data.length; i++) {
            var line = data[i].split(' ');
            var x = parseFloat(line[0]);
            var y = parseFloat(line[1]);
            var z = parseFloat(line[2]);
            geometry.vertices.push(
                new THREE.Vector3(x, y, z)
            );
        }

        var material =  new THREE.PointsMaterial( { color: green, size: pointsSize } );
        var points = new THREE.Points( geometry, material );
        scene.add(points);
    });

    $(".loadCompressedModel_CM").click(function(){

        clearScene();
        
        const file_path = '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/all_xyz_points3D_obvs_mean.txt';

        var data = fs.readFileSync(file_path);
        data = data.toString().split('\n');
        data.pop() // remove the last element ""

        console.log("loadCompressedModel_CM " + data.length);

        var geometry = new THREE.Geometry();

        for (var i = 0; i < data.length; i++) {
            var line = data[i].split(' ');
            var x = parseFloat(line[1]);
            var y = parseFloat(line[2]);
            var z = parseFloat(line[3]);
            geometry.vertices.push(
                new THREE.Vector3(x, y, z)
            );
        }

        var material =  new THREE.PointsMaterial( { color: blue, size: pointsSize } );
        var points = new THREE.Points( geometry, material );
        scene.add(points);
    });

    $(".loadQueryColmapPose").click(function(){

        var colmapPose = fs.readFileSync("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/query_pose.txt");
        colmapPose = colmapPose.toString().split('\n');

        var tx = parseFloat(colmapPose[4]);
        var ty = parseFloat(colmapPose[5]);
        var tz = parseFloat(colmapPose[6]);

        var qx = parseFloat(colmapPose[1]);
        var qy = parseFloat(colmapPose[2]);
        var qz = parseFloat(colmapPose[3]);
        var qw = parseFloat(colmapPose[0]);

        var pose_geometry = new THREE.ConeGeometry( 0.5, 0.75, 4 );
        var material = new THREE.MeshPhongMaterial( {color: yellow} );
        var pose_cam = new THREE.Mesh( pose_geometry, material );
        pose_cam.rotation.y = Math.PI/4 ;
        pose_cam.rotation.x = -Math.PI/2 ;
        scene.add( pose_cam );

        pose_cam.position.x = tx;
        pose_cam.position.y = ty;
        pose_cam.position.z = tz;

        var quaternion = new THREE.Quaternion();
        quaternion.fromArray([qx, qy, qz, qw]);
        quaternion.normalize(); // ?
        pose_cam.setRotationFromQuaternion(quaternion);

    });

    $(".loadColmapPoses").click(function(){

        var images_no = fs.readFileSync("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/images_no.txt");
        images_no = parseFloat(images_no.toString());

        for (var colmapPoseIndex = 1; colmapPoseIndex <= images_no; colmapPoseIndex++) {

            colmapPose = fs.readFileSync("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/pose_"+colmapPoseIndex+".txt");
            colmapPose = colmapPose.toString().split('\n');

            var tx = parseFloat(colmapPose[4]);
            var ty = parseFloat(colmapPose[5]);
            var tz = parseFloat(colmapPose[6]);

            var qx = parseFloat(colmapPose[1]);
            var qy = parseFloat(colmapPose[2]);
            var qz = parseFloat(colmapPose[3]);
            var qw = parseFloat(colmapPose[0]);

            var pose_geometry = new THREE.ConeGeometry( 0.5, 0.75, 4 );
            var material = new THREE.MeshPhongMaterial( {color: yellow} );
            var pose_cam = new THREE.Mesh( pose_geometry, material );
            pose_cam.rotation.y = Math.PI/4 ;
            pose_cam.rotation.x = -Math.PI/2 ;
            scene.add( pose_cam );

            pose_cam.position.x = tx;
            pose_cam.position.y = ty;
            pose_cam.position.z = tz;

            var quaternion = new THREE.Quaternion();
            quaternion.fromArray([qx, qy, qz, qw]);
            quaternion.normalize(); // ?
            pose_cam.setRotationFromQuaternion(quaternion);
            pose_cam.rotation.x = Math.PI;

            // const file_path = '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/points3D_'+colmapPoseIndex+'.txt';
            //
            // var data = fs.readFileSync(file_path);
            // data = data.toString().split('\n');
            //
            // var geometry = new THREE.Geometry();
            //
            // for (var i = 0; i < data.length; i++) {
            //     var line = data[i].split(' ');
            //     var x = parseFloat(line[0]);
            //     var y = parseFloat(line[1]);
            //     var z = parseFloat(line[2]);
            //     geometry.vertices.push(
            //         new THREE.Vector3(x, y, z)
            //     );
            // }
            //
            // var material =  new THREE.PointsMaterial( { color: green, size: pointsSize } );
            // var points = new THREE.Points( geometry, material );
            // // points.scale.set(0.1,0.1,0.1);
            // scene.add(points);
        }
    });

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

    var renderer = new THREE.WebGLRenderer({canvas: document.getElementById( "drawingSurface" )});
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );

    // var size = 10;
    // var divisions = 10;

    // var gridHelper = new THREE.GridHelper( size, divisions );
    // scene.add( gridHelper );
    //
    // var axesHelper = new THREE.AxesHelper( 5 );
    // scene.add( axesHelper );

    // lights
    var light = new THREE.DirectionalLight( white );
    var ambientLight = new THREE.AmbientLight( pink );
    light.position.set( 50, 50, 50 );
    scene.add( light );
    scene.add(ambientLight);

    controls = new THREE.TrackballControls(camera, renderer.domElement);

    camera.position.set( 1.927033026880825, 3.5235899349786655, -8.911491856699465);
    camera.lookAt(scene.position);

    controls.update(); //must be called after any manual changes to the camera's transform

    function animate() {
        requestAnimationFrame( animate );
        // required if controls.enableDamping or controls.autoRotate are set to true
        controls.update();
        renderer.render( scene, camera );
    }

    animate();
};

function clearScene(){
    while(scene.children.length > 0){
        scene.remove(scene.children[0]);
    }
}

function renderModel(i, colour) {
    var file_path = '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/'+i+'/all_xyz_points3D.txt';
    var data = fs.readFileSync(file_path);
    data = data.toString().split('\n');

    console.log(data.length-1);
    totalPointsSum += data.length-1;

    var geometry = new THREE.Geometry();

    for (var i = 0; i < data.length; i++) {
        var line = data[i].split(' ');
        var x = parseFloat(line[1]);
        var y = parseFloat(line[2]);
        var z = parseFloat(line[3]);
        geometry.vertices.push(
            new THREE.Vector3(x, y, z)
        );
    }

    var material =  new THREE.PointsMaterial( { color: colour, size: pointsSize } );
    var points = new THREE.Points( geometry, material );
    scene.add(points);
}

// function get3DPoints(){
//     console.log("Getting 3D points from COLMAP");
//     execSync('cd /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/ && python3 create_3D_points_for_ARCore_debug.py');
// }

// function read3Dpoints(){
//
//     scene.remove(colmap_points); // remove previous ones
//
//     const file_path = '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/points3D_AR.txt';
//
//     var data = fs.readFileSync(file_path);
//     data = data.toString().split('\n');
//
//     var geometry = new THREE.Geometry();
//
//     for (var i = 0; i < data.length; i++) {
//         xyz = data[i].split(' ');
//         x = parseFloat(xyz[0]);
//         y = parseFloat(xyz[1]);
//         z = parseFloat(xyz[2]);
//         geometry.vertices.push(
//             new THREE.Vector3(x, y, z)
//         )
//     }
//
//     var material =  new THREE.PointsMaterial( { color: red, size: 0.02 } );
//     colmap_points = new THREE.Points( geometry, material );
//
//     colmap_points.rotation.z = Math.PI/2; // is this needed ? (bug) ?
//     scene.add(colmap_points);
// }

// function getModel(){
//     const file_path = '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/all_points3D.txt';
//
//     var data = fs.readFileSync(file_path);
//     data = data.toString().split('\n');
//
//     return data;
// }

// function loadPoints3DFromFile(){
//     const file_path = '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/points3D_AR.txt';
//
//     var data = fs.readFileSync(file_path);
//     data = data.toString().split('\n');
//
//     return data;
// }

// function localise(arg_pose, arg_pose_matrix){
//
//     var pose = arg_pose;
//     var pose_matrix_string = arg_pose_matrix;
//     //server.close();
//
//     var base64String = $('.frame').attr('src');
//     var base64Data = base64String.replace(/^data:image\/png;base64,/, "");
//
//     fs.writeFileSync("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/current_query_image/query.jpg",
//         base64Data, 'base64', function(err) {
//             console.log(err);
//         });
//
//     fs.writeFileSync("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_data/cameraPose.txt",
//         pose.join(","), function(err) {
//             console.log(err);
//         });
//
//     fs.writeFileSync("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_data/cameraPoseMatrixString.txt",
//         pose_matrix_string, function(err) {
//             console.log(err);
//         });
//
//     //rotate image so it matches the ones in COLMAP
//     // execSync('sips -r 90 /Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/current_query_image/query.jpg')
//
//     //remove model and replace with vanilla
//     execSync('rm -rf /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/model/');
//     execSync('rm /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/database.db');
//     execSync('cp -r /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/vanilla_model/* /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/');
//
//     //remove old localised model
//     execSync('rm -rf /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/new_model/*');
//     execSync('cd /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/ && python3 register_query_image.py');
//
//     console.log('Done localising!');
//
//     execSync('cd /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/ && python3 debug_results.py');
//     $(".colmap_result_frame").attr('src', '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/colmap_points_projected.jpg');
//
//     var global_pose = execSync('cd /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/ && python3 get_global_pose.py');
//
//     return global_pose.toString();
// }
//
// function exportMatrixString(matrix, name){
//     fs.writeFileSync("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/"+name+".txt",
//         matrix, 'utf8', function(err) {
//             console.log(err);
//         });
// }
//
// function exportARCorePointCloud() {
//
//     var arcore_points_String = "";
//     for (var i = 0; i < arcore_points.geometry.vertices.length; i++) {
//
//         var x = arcore_points.geometry.vertices[i].x;
//         var y = arcore_points.geometry.vertices[i].y;
//         var z = arcore_points.geometry.vertices[i].z;
//
//         arcore_points_String += x + " " + y + " " + z + " " + 1 +"\n"
//     }
//
//     fs.writeFileSync("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/arcore_pointCloud.txt",
//         arcore_points_String, 'utf8', function(err) {
//             console.log(err);
//         });
// }
//
// function debug_COLMAP_points(scale){
//     execSync('cd /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/ && python3 create_3D_points_for_ARCore_debug.py ' + scale);
//     read3Dpoints();
// }
