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
var pointsSize = 0.1;
var pose_scale = 0.3
var base_model_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/tmp/points3D.txt"
var cameras_data_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/tmp/camera_centers.txt"
var CMU_gt_session = "session_2"
var gt_poses_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/tmp/gt_query_data.txt"
var gt_images_names = []

window.onload = function() {

    var handle = $( "#custom-handle" );
    $( "#slider" ).slider({
        value: 100,
        min: 1,
        max: 100,
        create: function() {
            handle.text( $( this ).slider( "value" ) + "%");
        },
        slide: function( event, ui ) {
            percentage = ui.value
            handle.text( ui.value + "%");
            renderModelPath("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/points3D_sorted_descending.txt", red, percentage);
        }});

    $(".load_sorted_points").click(function(){
        renderModelPath("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/threejs_data_exported/points3D_sorted_descending.txt", red);
    });

    $(".reset").click(function(){
        clearScene();
    });

    $(".loadBaseModel").click(function(){
        const file_path = base_model_path;
        var data = fs.readFileSync(file_path);
        data = data.toString().split('\n');
        data.pop() //get last "" out

        // so model is centered
        var sum_x = 0
        var sum_y = 0
        var sum_z = 0
        for (var i = 3; i < data.length; i++) {
            var data_line = data[i].split(" ")
            var x = parseFloat(data_line[1]);
            var y = parseFloat(data_line[2]);
            var z = parseFloat(data_line[3]);

            sum_x += x;
            sum_y += y;
            sum_z += z;
        }
        var mean_x = sum_x / data.length;
        var mean_y = sum_y / data.length;
        var mean_z = sum_z / data.length;

        var geometry = new THREE.Geometry();

        for (var i = 3; i < data.length; i++) {

            var data_line = data[i].split(" ")

            var x = parseFloat(data_line[1]) - mean_x;
            var y = parseFloat(data_line[2]) - mean_y;
            var z = parseFloat(data_line[3]) - mean_z;

            geometry.vertices.push(
                new THREE.Vector3( x, y, z )
            );

            var r = parseInt(data_line[4])
            var g = parseInt(data_line[5])
            var b = parseInt(data_line[6])

            geometry.colors.push(
                new THREE.Color( "rgb("+r+","+g+","+b+")" )
            );
        }

        var material = new THREE.PointsMaterial( { size: pointsSize, vertexColors: THREE.VertexColors } );
        var points = new THREE.Points( geometry, material );
        scene.add( points );
        console.log("Done")
    });

    $(".loadBasePoses").click(function(){
        //before this run python3 get_camera_centers_from_images_to_file.py
        var pose_data = fs.readFileSync(cameras_data_path).toString().split('\n');
        pose_data.pop() //get last "" out

        // so poses are centered like the model (have to use the model here)
        const file_path = base_model_path;
        var data = fs.readFileSync(file_path);
        data = data.toString().split('\n');
        data.pop() //get last "" out

        // so model is centered
        var sum_x = 0
        var sum_y = 0
        var sum_z = 0
        for (var i = 3; i < data.length; i++) {
            var data_line = data[i].split(" ")
            var x = parseFloat(data_line[1]);
            var y = parseFloat(data_line[2]);
            var z = parseFloat(data_line[3]);

            sum_x += x;
            sum_y += y;
            sum_z += z;
        }
        var mean_x = sum_x / data.length;
        var mean_y = sum_y / data.length;
        var mean_z = sum_z / data.length;

        for (var i = 0; i < pose_data.length; i++) {
            var data = pose_data[i].split(" ")

            var tx = parseFloat(data[0]) - mean_x;
            var ty = parseFloat(data[1]) - mean_y;
            var tz = parseFloat(data[2]) - mean_z;

            var qx = parseFloat(data[3]);
            var qy = parseFloat(data[4]);
            var qz = parseFloat(data[5]);
            var qw = parseFloat(data[6]);

            //principal vector axis
            var pva_x = parseFloat(data[7]);
            var pva_y = parseFloat(data[8]);
            var pva_z = parseFloat(data[9]);

            var color = blue;
            var session = data[10].split("/")[0];

            console.log(session);

            if (session == CMU_gt_session) {
                gt_images_names.push(data[10].split("/")[1])
                color = green;
            }
            else{
                continue
            }

            if(session.startsWith("img")){
                color = red;
            }
            // if (session == "session_10") {
            //     color = blue
            // }

            var pose_geometry = new THREE.SphereGeometry(  0.5, 8, 8  );
            var material = new THREE.MeshPhongMaterial( {color: color} );
            var pose_cam = new THREE.Mesh( pose_geometry, material );

            scene.add( pose_cam );

            //principal vector axes
            var line_points = [];
            var temp_scale = 150000;
            line_points.push( new THREE.Vector3( tx, ty, tz ) );
            line_points.push( new THREE.Vector3( tx + pva_x / temp_scale, ty + pva_y / temp_scale, tz + pva_z / temp_scale ) );

            var line_material = new THREE.LineBasicMaterial( { color: white } );
            var line_geometry = new THREE.BufferGeometry().setFromPoints( line_points );
            var line = new THREE.Line( line_geometry, line_material );

            scene.add( line );

            var quaternion = new THREE.Quaternion();
            quaternion.fromArray([qx, qy, qz, qw]);
            pose_cam.setRotationFromQuaternion(quaternion);

            pose_cam.position.x = tx;
            pose_cam.position.y = ty;
            pose_cam.position.z = tz;

            pose_cam.scale.set(pose_scale,pose_scale,pose_scale);
        }

    });

    $(".loadGTPoses").click(function(){
        //before this run python3 get_principal_axis_vects.py
        var pose_data = fs.readFileSync(gt_poses_path).toString().split('\n');
        pose_data.pop() //get last "" out

        var color = yellow;

        // so poses are centered like the model (have to use the model here)
        const file_path = base_model_path;
        var data = fs.readFileSync(file_path);
        data = data.toString().split('\n');
        data.pop() //get last "" out

        // so model is centered
        var sum_x = 0
        var sum_y = 0
        var sum_z = 0
        for (var i = 3; i < data.length; i++) {
            var data_line = data[i].split(" ")
            var x = parseFloat(data_line[1]);
            var y = parseFloat(data_line[2]);
            var z = parseFloat(data_line[3]);

            sum_x += x;
            sum_y += y;
            sum_z += z;
        }
        var mean_x = sum_x / data.length;
        var mean_y = sum_y / data.length;
        var mean_z = sum_z / data.length;

        for (var i = 0; i < pose_data.length; i++) {

            var data = pose_data[i].split(" ");

            var name = data[0]

            if(gt_images_names.includes(name) == false){
                continue
            }

            var tx = parseFloat(data[5]) - mean_x;
            var ty = parseFloat(data[6]) - mean_y;
            var tz = parseFloat(data[7]) - mean_z;

            var qx = parseFloat(data[1]);
            var qy = parseFloat(data[2]);
            var qz = parseFloat(data[3]);
            var qw = parseFloat(data[4]);

            //principal vector axis
            var pva_x = parseFloat(data[8]);
            var pva_y = parseFloat(data[9]);
            var pva_z = parseFloat(data[10]);

            var pose_geometry = new THREE.SphereGeometry(  0.5, 8, 8  );
            var material = new THREE.MeshPhongMaterial( {color: color} );
            var pose_cam = new THREE.Mesh( pose_geometry, material );

            scene.add( pose_cam );

            //principal vector axes
            var line_points = [];
            var temp_scale = 170000;
            line_points.push( new THREE.Vector3( tx, ty, tz ) );
            line_points.push( new THREE.Vector3( tx + pva_x / temp_scale, ty + pva_y / temp_scale, tz + pva_z / temp_scale ) );

            var line_material = new THREE.LineBasicMaterial( { color: white } );
            var line_geometry = new THREE.BufferGeometry().setFromPoints( line_points );
            var line = new THREE.Line( line_geometry, line_material );

            scene.add( line );

            var quaternion = new THREE.Quaternion();
            quaternion.fromArray([qx, qy, qz, qw]);
            pose_cam.setRotationFromQuaternion(quaternion);

            pose_cam.position.x = tx;
            pose_cam.position.y = ty;
            pose_cam.position.z = tz;

            pose_cam.scale.set(pose_scale,pose_scale,pose_scale);
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
    var axesHelper = new THREE.AxesHelper( 5 );
    scene.add( axesHelper );

    // lights
    var light = new THREE.DirectionalLight( white );
    var ambientLight = new THREE.AmbientLight( pink );
    light.position.set( 5, 5, 5 );
    scene.add( light );
    scene.add(ambientLight);

    controls = new THREE.TrackballControls(camera, renderer.domElement);

    camera.position.set( 25, 25, 25);
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

function renderModelPath(file_path, colour, percentage=100) {
    clearScene()
    var data = fs.readFileSync(file_path);
    data = data.toString().split('\n');

    var geometry = new THREE.Geometry();

    len = Math.round(data.length * percentage / 100) - 1 //remove that last new line

    for (var i = 0; i <= len; i++) {
        var line = data[i].split(' ');
        var x = parseFloat(line[0]);
        var y = parseFloat(line[1]);
        var z = parseFloat(line[2]);
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
