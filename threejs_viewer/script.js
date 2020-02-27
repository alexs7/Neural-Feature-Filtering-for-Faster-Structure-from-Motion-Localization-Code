const fs = require('fs');
const express = require('express');
const bodyParser = require('body-parser');
const { execSync } = require('child_process');

//3D Objects
var phone_cam_0; // displayOrientedMode
var phone_cam_1; // cameraPose
var anchor;
var colmap_points;
var arcore_points;
var scene;
var server;
var cameraPose;
var cameraDisplayOrientedPose;
var camerasOffset = 0.0;
var camera; //ThreeJS Camera
var controls;

window.onload = function() {

    //assign button listeners
    $(".resetCamera").click(function(){
        camera.position.set( 0.1, 1, 1 );
        camera.lookAt(scene.position);
        controls.update();
    });

    $(".exportPoints").click(function(){
        exportPoints();
    });

    $(".loadCOLMAPpoints").click(function(){
        read3Dpoints();
        console.log("Done loading points");
    });

    $(".viewARCoreCame").click(function(){

        tx = parseFloat(cameraDisplayOrientedPose[0]);
        ty = parseFloat(cameraDisplayOrientedPose[1]);
        tz = parseFloat(cameraDisplayOrientedPose[2]);
        qx = parseFloat(cameraDisplayOrientedPose[3]);
        qy = parseFloat(cameraDisplayOrientedPose[4]);
        qz = parseFloat(cameraDisplayOrientedPose[5]);
        qw = parseFloat(cameraDisplayOrientedPose[6]);

        phone_cam_1.position.x = tx;
        phone_cam_1.position.y = ty;
        phone_cam_1.position.z = tz ;

        var rotMatrix = new THREE.Matrix4();
        var quat = new THREE.Quaternion();
        quat.fromArray([qx,qy,qz,qw]);
        quat.normalize();
        rotMatrix.makeRotationFromQuaternion(quat);
        rotMatrix.setPosition(tx,ty,tz);

        var lookAtPoint =  new THREE.Vector4([0, 0, -1, 1]);
        lookAtPoint.applyMatrix4(rotMatrix);

        debugger;
        phone_cam_1.lookAt(0,0,0);

        //controls.update();
    });

    $( ".localiseButton" ).click(function() {
        localise();
    });

    //start server
    const app = express();
    app.use(bodyParser.urlencoded({ extended: true, limit: '1mb' }));
    app.use(bodyParser.json({limit: '1mb'}));

    app.post('/', (req, res) => {
        $(".frame").attr('src', 'data:image/png;base64,'+req.body.frameString);

        cameraDisplayOrientedPose = req.body.cameraDisplayOrientedPose.split(',');

        tx = parseFloat(cameraDisplayOrientedPose[0]);
        ty = parseFloat(cameraDisplayOrientedPose[1]);
        tz = parseFloat(cameraDisplayOrientedPose[2]);
        qx = parseFloat(cameraDisplayOrientedPose[3]);
        qy = parseFloat(cameraDisplayOrientedPose[4]);
        qz = parseFloat(cameraDisplayOrientedPose[5]);
        qw = parseFloat(cameraDisplayOrientedPose[6]);

        phone_cam_0.position.x = tx - camerasOffset;
        phone_cam_0.position.y = ty;
        phone_cam_0.position.z = tz;

        var quaternion = new THREE.Quaternion();
        quaternion.fromArray([qx,qy,qz,qw]);
        quaternion.normalize();
        phone_cam_0.setRotationFromQuaternion(quaternion);

        // cameraPose = req.body.cameraPose.split(',');
        //
        // tx = parseFloat(cameraPose[0]);
        // ty = parseFloat(cameraPose[1]);
        // tz = parseFloat(cameraPose[2]);
        // qx = parseFloat(cameraPose[3]);
        // qy = parseFloat(cameraPose[4]);
        // qz = parseFloat(cameraPose[5]);
        // qw = parseFloat(cameraPose[6]);
        //
        // phone_cam_1.position.x = tx + camerasOffset;
        // phone_cam_1.position.y = ty;
        // phone_cam_1.position.z = tz;
        //
        // var quaternion = new THREE.Quaternion();
        // quaternion.fromArray([qx,qy,qz,qw]);
        // quaternion.normalize();
        // phone_cam_1.setRotationFromQuaternion(quaternion);

        var anchorPosition = req.body.anchorPosition.split(',');
        anchor_tx = parseFloat(anchorPosition[0]);
        anchor_ty = parseFloat(anchorPosition[1]);
        anchor_tz = parseFloat(anchorPosition[2]);

        anchor.position.x = anchor_tx;
        anchor.position.y = anchor_ty;
        anchor.position.z = anchor_tz;

        pointsArray = req.body.pointCloud.split("\n");
        pointsArray.pop();

        scene.remove(arcore_points);
        var pointsGeometry = new THREE.Geometry();
        var material =  new THREE.PointsMaterial( { color: 0x00ff00, size: 0.008 } );

        for (var i = 0; i < pointsArray.length; i++) {
            x = parseFloat(pointsArray[i].split(" ")[0]);
            y = parseFloat(pointsArray[i].split(" ")[1]);
            z = parseFloat(pointsArray[i].split(" ")[2]);

            pointsGeometry.vertices.push(
                new THREE.Vector3(x, y, z)
            )
        }

        arcore_points = new THREE.Points( pointsGeometry, material );
        scene.add(arcore_points);

        res.sendStatus(200);
    });

    app.post('/localise', (req, res) => {
        localise();
    });

    server = app.listen(3000, () => console.log(`Started server at http://localhost:3000!`));

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

    var renderer = new THREE.WebGLRenderer({canvas: document.getElementById( "drawingSurface" )});
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );

    var size = 10;
    var divisions = 10;

    var gridHelper = new THREE.GridHelper( size, divisions );
    scene.add( gridHelper );

    var axesHelper = new THREE.AxesHelper( 5 );
    scene.add( axesHelper );

    var geometry = new THREE.Geometry();
    geometry.vertices.push(
        new THREE.Vector3(1, 1.4, 0),
        new THREE.Vector3(0.5, 0.9, 0),
        new THREE.Vector3(-1, 1.4, 0),
        new THREE.Vector3(-0.5, 0.9, 0),
        new THREE.Vector3(-1, -1.4, 0),
        new THREE.Vector3(-0.5, -0.9, 0),
        new THREE.Vector3(1, -1.4, 0),
        new THREE.Vector3(0.5, -0.9, 0),
        new THREE.Vector3(0, 0, 1),
        new THREE.Vector3(0, 0, 1.5),
        new THREE.Vector3(0, 0, 2),
        new THREE.Vector3(0, 0, 2.5),
        new THREE.Vector3(0, 0, 3)
    );

    var material =  new THREE.PointsMaterial( { color: 0xff0000, size: 0.03 } );
    phone_cam_0 = new THREE.Points( geometry, material );
    phone_cam_0.position.z = camerasOffset;
    phone_cam_0.position.x = -camerasOffset;

    var axesHelperCone = new THREE.AxesHelper( 2 );
    phone_cam_0.add(axesHelperCone);
    phone_cam_0.scale.set(0.1,0.1,0.1);
    scene.add( phone_cam_0 );

    var material =  new THREE.PointsMaterial( { color: 0x9900FF, size: 0.03 } );
    phone_cam_1 = new THREE.Points( geometry, material );
    phone_cam_1.position.z = camerasOffset;
    phone_cam_1.position.x = camerasOffset;

    var axesHelperCone = new THREE.AxesHelper( 2 );
    phone_cam_1.add(axesHelperCone);
    phone_cam_1.scale.set(0.1,0.1,0.1);
    //scene.add( phone_cam_1 );

    var geometry = new THREE.SphereGeometry( 1, 32, 32 );
    var material = new THREE.MeshPhongMaterial( {color: 0xffff00} );
    anchor = new THREE.Mesh( geometry, material );
    scene.add( anchor );
    anchor.scale.set(0.03,0.03,0.03);

    //reference points
    var geometry = new THREE.SphereGeometry( 1, 32, 32 );
    var material = new THREE.MeshPhongMaterial( {color: 0x00FFFF} );
    var reference_point_1 = new THREE.Mesh( geometry, material );
    scene.add( reference_point_1 );
    reference_point_1.scale.set(0.04,0.04,0.04);
    reference_point_1.position.set(-0.5,0,-1);

    var geometry = new THREE.SphereGeometry( 1, 32, 32 );
    var material = new THREE.MeshPhongMaterial( {color: 0xFD00FF} );
    var reference_point_2 = new THREE.Mesh( geometry, material );
    scene.add( reference_point_2 );
    reference_point_2.scale.set(0.04,0.04,0.04);
    reference_point_2.position.set(0.5,0,-1);

    // lights
    var light = new THREE.DirectionalLight( 0xffffff );
    var ambientLight = new THREE.AmbientLight( 0x404040 );
    light.position.set( 50, 50, 50 );
    scene.add( light );
    scene.add(ambientLight);

    controls = new THREE.OrbitControls(camera, renderer.domElement);

    camera.position.set( 0.1, 1, 1 );
    camera.lookAt(scene.position);

    controls.update(); //must be called after any manual changes to the camera's transform

    function animate() {
        requestAnimationFrame( animate );
        // required if controls.enableDamping or controls.autoRotate are set to true
        controls.update();
        renderer.render( scene, camera );
    }

    animate();

    $( ".slider_size" ).slider({
        min: 0.01,
        max: 0.03,
        step: 0.0001,
        slide: function( event, ui ) {
            var size = ui.value;
            colmap_points.material.size = size;
        }
    });

    $( ".slider_scale" ).slider({
        min: 0.01,
        max: 1,
        step: 0.005,
        slide: function( event, ui ) {
            var scale = ui.value;
            colmap_points.scale.set(scale,scale,scale);
        }
    });
};

function read3Dpoints(){

    scene.remove(colmap_points); // remove previous ones

    const file_path = '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/points3D_AR.txt';
    // async !
    var data = fs.readFileSync(file_path);
    data = data.toString().split('\n');

    var geometry = new THREE.Geometry();

    for (var i = 0; i < data.length; i++) {
        xyz = data[i].split(' ');
        x = parseFloat(xyz[0]);
        y = parseFloat(xyz[1]);
        z = parseFloat(xyz[2]);
        geometry.vertices.push(
            new THREE.Vector3(x, y, z)
        )
    }

    var material =  new THREE.PointsMaterial( { color: 0x0000ff, size: 0.005 } );
    colmap_points = new THREE.Points( geometry, material );

    scene.add(colmap_points);
}

function localise(){
    server.close();

    var base64String = $('.frame').attr('src');
    var base64Data = base64String.replace(/^data:image\/png;base64,/, "");

    fs.writeFileSync("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/current_query_image/query.jpg",
        base64Data, 'base64', function(err) {
            console.log(err);
        });

    fs.writeFileSync("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_data/cameraPose.txt",
        cameraDisplayOrientedPose.join(","), function(err) {
            console.log(err);
        });

    //remove model and replace with vanilla
    execSync('rm -rf /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/model/');
    execSync('rm /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/database.db');
    execSync('cp -r /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/vanilla_model/* /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/');

    execSync('rm -rf /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/new_model/*');
    execSync('cd /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/ && python3 register_query_image.py');

    console.log('Done localising!');

    execSync('cd /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/ && python3 debug_results.py');
    $(".colmap_result_frame").attr('src', '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/colmap_points_projected.jpg');

    // console.log("Loading 3D points now!");
    // execSync('cd /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/ && python3 create_3D_points_for_ARCore_debug.py');
    // read3Dpoints();
}

function exportPoints() {

    var arcore_points_String = "";
    for (var i = 0; i < arcore_points.geometry.vertices.length; i++) {

        var x = arcore_points.geometry.vertices[i].x;
        var y = arcore_points.geometry.vertices[i].y;
        var z = arcore_points.geometry.vertices[i].z;

        arcore_points_String += x + " " + y + " " + z + "\n"
    }

    fs.writeFileSync("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/arcore_points.txt",
        arcore_points_String, 'utf8', function(err) {
            console.log(err);
        });

    var colmap_points_String = "";
    for (var i = 0; i < colmap_points.geometry.vertices.length; i++) {

        var x = colmap_points.geometry.vertices[i].x;
        var y = colmap_points.geometry.vertices[i].y;
        var z = colmap_points.geometry.vertices[i].z;

        colmap_points_String += x + " " + y + " " + z + "\n"
    }

    fs.writeFileSync("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/colmap_points.txt",
        colmap_points_String, 'utf8', function(err) {
            console.log(err);
        });
}