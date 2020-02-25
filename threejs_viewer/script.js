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

window.onload = function() {

    //assign button listeners
    $( ".localiseButton" ).click(function() {

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

        execSync('rm -rf /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/new_model/*');
        execSync('cd /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/ && python3 register_query_image.py');

        console.log('Done localising!');

        console.log('Set camera position to ARCore camera');

        camera.position.x = phone_cam_0.position.x;
        camera.position.y = phone_cam_0.position.y;
        camera.position.z = phone_cam_0.position.z;

        rotObjectMatrix = new THREE.Matrix4();
        rotObjectMatrix.makeRotationFromQuaternion(phone_cam_0.quaternion);
        camera.quaternion.setFromRotationMatrix(rotObjectMatrix);

        console.log("Loading 3D points now!");
        // execSync('cd /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/ && python3 create_3D_points_for_ARCore_debug.py');
        // read3Dpoints();

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
        new THREE.Vector3(1, 1, 0),
        new THREE.Vector3(0.5, 0.5, 0),
        new THREE.Vector3(-1, 1, 0),
        new THREE.Vector3(-0.5, 0.5, 0),
        new THREE.Vector3(-1, -1, 0),
        new THREE.Vector3(-0.5, -0.5, 0),
        new THREE.Vector3(1, -1, 0),
        new THREE.Vector3(0.5, -0.5, 0),
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
    // scene.add( phone_cam_1 );

    var geometry = new THREE.SphereGeometry( 1, 32, 32 );
    var material = new THREE.MeshPhongMaterial( {color: 0xffff00} );
    anchor = new THREE.Mesh( geometry, material );
    scene.add( anchor );
    anchor.scale.set(0.03,0.03,0.03);

    var light = new THREE.DirectionalLight( 0xffffff );
    light.position.set( 50, 50, 50 );
    scene.add( light );

    var controls = new THREE.TrackballControls(camera, renderer.domElement);

    //controls.update() must be called after any manual changes to the camera's transform
    camera.position.set( 1, 1, 1 );
    camera.lookAt(scene.position);
    // controls.update();

    function animate() {
        requestAnimationFrame( animate );
        // required if controls.enableDamping or controls.autoRotate are set to true
        // controls.update();
        renderer.render( scene, camera );
    }

    animate();

    $( ".slider" ).slider({
        min: 0.01,
        max: 0.1,
        step: 0.005,
        slide: function( event, ui ) {
            var scale = ui.value;
            colmap_points.scale.set(scale,scale,scale);
        }
    });
};

function read3Dpoints(){

    const file_path = '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/points3D_AR.txt';
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

    var material =  new THREE.PointsMaterial( { color: 0x0000ff, size: 0.001 } );
    colmap_points = new THREE.Points( geometry, material );

    scene.add(colmap_points);
}