const fs = require('fs');
const express = require('express');
const bodyParser = require('body-parser');

//3D Objects
var cone;
var anchor;
var x_cam_point_axis;
var y_cam_point_axis;
var z_cam_point_axis;
var points;
var scene;

window.onload = function() {

    //start server
    const app = express();
    app.use(bodyParser.urlencoded({ extended: true }));
    app.use(bodyParser.text());

    app.post('/', (req, res) => {
        // console.log('Got body:', req.body);

        tx = parseFloat(req.body.split(',')[0]);
        ty = parseFloat(req.body.split(',')[1]);
        tz = parseFloat(req.body.split(',')[2]);
        qx = parseFloat(req.body.split(',')[3]);
        qy = parseFloat(req.body.split(',')[4]);
        qz = parseFloat(req.body.split(',')[5]);
        qw = parseFloat(req.body.split(',')[6]);

        cone.position.x = tx;
        cone.position.y = ty;
        cone.position.z = tz;

        x_cam_point_axis_xyz = JSON.parse(req.body.split("|")[1]);
        x_cam_point_axis.position.x = x_cam_point_axis_xyz[0];
        x_cam_point_axis.position.y = x_cam_point_axis_xyz[1];
        x_cam_point_axis.position.z = x_cam_point_axis_xyz[2];

        y_cam_point_axis_xyz = JSON.parse(req.body.split("|")[2]);
        y_cam_point_axis.position.x = y_cam_point_axis_xyz[0];
        y_cam_point_axis.position.y = y_cam_point_axis_xyz[1];
        y_cam_point_axis.position.z = y_cam_point_axis_xyz[2];

        z_cam_point_axis_xyz = JSON.parse(req.body.split("|")[3]);
        z_cam_point_axis.position.x = z_cam_point_axis_xyz[0];
        z_cam_point_axis.position.y = z_cam_point_axis_xyz[1];
        z_cam_point_axis.position.z = z_cam_point_axis_xyz[2];

        var quaternion = new THREE.Quaternion();
        quaternion.fromArray([qx,qy,qz,qw]);
        quaternion.normalize();
        cone.setRotationFromQuaternion(quaternion);

        // cone.rotation.x += Math.PI/2;

        anchor_tx = parseFloat(req.body.split(',')[7]);
        anchor_ty = parseFloat(req.body.split(',')[8]);
        anchor_tz = parseFloat(req.body.split(',')[9]);

        anchor.position.x = anchor_tx;
        anchor.position.y = anchor_ty;
        anchor.position.z = anchor_tz;

        res.sendStatus(200);
    });

    app.listen(3000, () => console.log(`Started server at http://localhost:3000!`));

    scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

    var renderer = new THREE.WebGLRenderer();
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );

    var size = 10;
    var divisions = 10;

    var gridHelper = new THREE.GridHelper( size, divisions );
    scene.add( gridHelper );

    var axesHelper = new THREE.AxesHelper( 5 );
    scene.add( axesHelper );

    // var geometry = new THREE.ConeGeometry( 1, 1, 4 );

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
    cone = new THREE.Points( geometry, material );
    cone.position.z = 0.25;

    var axesHelperCone = new THREE.AxesHelper( 5 );
    cone.add(axesHelperCone);
    scene.add( cone );
    cone.scale.set(0.1,0.1,0.1);

    var geometry = new THREE.SphereGeometry( 1, 32, 32 );
    var material = new THREE.MeshPhongMaterial( {color: 0xffff00} );
    anchor = new THREE.Mesh( geometry, material );
    scene.add( anchor );
    anchor.scale.set(0.05,0.05,0.05);

    var geometry = new THREE.SphereGeometry( 1, 32, 32 );
    var material = new THREE.MeshPhongMaterial( {color: 'red'} );
    x_cam_point_axis = new THREE.Mesh( geometry, material );
    scene.add( x_cam_point_axis );
    x_cam_point_axis.scale.set(0.025,0.025,0.025);

    var geometry = new THREE.SphereGeometry( 1, 32, 32 );
    var material = new THREE.MeshPhongMaterial( {color: 'green'} );
    y_cam_point_axis = new THREE.Mesh( geometry, material );
    scene.add( y_cam_point_axis );
    y_cam_point_axis.scale.set(0.025,0.025,0.025);

    var geometry = new THREE.SphereGeometry( 1, 32, 32 );
    var material = new THREE.MeshPhongMaterial( {color: 'blue'} );
    z_cam_point_axis = new THREE.Mesh( geometry, material );
    scene.add( z_cam_point_axis );
    z_cam_point_axis.scale.set(0.025,0.025,0.025);

    var light = new THREE.DirectionalLight( 0xffffff );
    light.position.set( 50, 50, 50 );
    scene.add( light );

    var controls = new THREE.TrackballControls(camera, renderer.domElement);

    //controls.update() must be called after any manual changes to the camera's transform
    camera.position.set( 1, 1, 1 );
    controls.update();

    function animate() {
        requestAnimationFrame( animate );
        // required if controls.enableDamping or controls.autoRotate are set to true
        controls.update();
        renderer.render( scene, camera );
    }

    animate();
};

function read3Dpoints(){

    const file_path = '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/points3D_AR.txt';
    fs.readFile(file_path, 'utf8', (err, data) => {

        var geometry = new THREE.Geometry();
        data = data.split('\n');

        for (var i = 0; i < data.length; i++) {
            xyz = data[i].split(' ');
            x = parseFloat(xyz[0]);
            y = parseFloat(xyz[1]);
            z = parseFloat(xyz[2]);
            geometry.vertices.push(
                new THREE.Vector3(x, y, z)
            )
        }

        var material =  new THREE.PointsMaterial( { color: 0x00ff00, size: 0.03 } );
        points = new THREE.Points( geometry, material );
        scene.add(points);
    });
}