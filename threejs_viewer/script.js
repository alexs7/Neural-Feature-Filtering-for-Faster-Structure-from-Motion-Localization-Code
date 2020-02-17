const fs = require('fs');
const express = require('express');
const bodyParser = require('body-parser');
const hostname = '127.0.0.1';
const port = 3000;

//3D Objects
var cone;

window.onload = function() {

    //start server
    const app = express();
    app.use(bodyParser.urlencoded({ extended: true }));
    app.use(bodyParser.text());

    app.post('/', (req, res) => {
        // console.log('Got body:', req.body);

        res.sendStatus(200);
    });
    
    app.listen(3000, () => console.log(`Started server at http://localhost:3000!`));

    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

    var renderer = new THREE.WebGLRenderer();
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );

    var size = 100;
    var divisions = 100;

    var gridHelper = new THREE.GridHelper( size, divisions );
    scene.add( gridHelper );

    var axesHelper = new THREE.AxesHelper( 5 );
    scene.add( axesHelper );

    var geometry = new THREE.ConeGeometry( 1, 1, 4 );
    var wireframe = new THREE.WireframeGeometry( geometry );
    cone = new THREE.LineSegments( wireframe );
    cone.material.color.setHex( 0xff0000 );
    cone.rotation.x = Math.PI/2;
    cone.rotation.y = Math.PI/4;
    cone.position.z = 0.5;
    scene.add( cone );

    var controls = new THREE.TrackballControls(camera, renderer.domElement);

    //controls.update() must be called after any manual changes to the camera's transform
    camera.position.set( 5, 5, 5 );
    controls.update();

    function animate() {
        requestAnimationFrame( animate );
        // required if controls.enableDamping or controls.autoRotate are set to true
        controls.update();
        renderer.render( scene, camera );
    }

    // //add poses here
    // const global_poses = '../global_poses/';
    //
    // fs.readdirSync(global_poses).forEach(file => {
    //     if(file == ".DS_Store"){
    //         return;
    //     }
    //     fs.readFile(global_poses+file, 'utf8', (err, data) => {
    //         console.log(file);
    //         // debugger;
    //         data = data.split("\n").slice(0,4);
    //         data = data.join(" ");
    //         data = data.split(" ");
    //         data = data.map(Number);
    //
    //         pose = new THREE.Matrix4().fromArray(data);
    //
    //         var geometry = new THREE.ConeGeometry( 1, 1, 4 );
    //         var edges = new THREE.EdgesGeometry(geometry);
    //         var cone = new THREE.LineSegments( edges, new THREE.LineBasicMaterial( { color: "red" } ) );
    //
    //         cone.setRotationFromMatrix(pose);
    //         cone.translateX(pose.elements[3]);
    //         cone.translateY(pose.elements[7]);
    //         cone.translateZ(pose.elements[11]);
    //         cone.scale.x = 0.08;
    //         cone.scale.y = 0.08;
    //         cone.scale.z = 0.08;
    //
    //         // scene.updateMatrixWorld();
    //         scene.add( cone );
    //     });
    // });

    animate();
};