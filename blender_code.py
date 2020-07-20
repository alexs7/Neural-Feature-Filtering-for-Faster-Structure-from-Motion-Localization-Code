import bpy
import mathutils
import numpy as np
import glob
import pdb
import collections

"""
1.delete all objects
"""
bpy.ops.object.select_all(action='DESELECT')  # Deselect all
for ob in bpy.context.scene.objects:
    ob.select = True
    bpy.ops.object.delete()

# bpy.ops.import_mesh.ply(
#     filepath="/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/model/0/model.ply")
# store = bpy.context.active_object

for ob in bpy.context.selected_objects:
    ob.select = False


# make sure you run "evaluator.py" first

def get_poses_from_dir(dir):
    poses = []
    for filename in glob.glob(dir + "/" + 'pose_*.txt'):
        poses.append(np.loadtxt(filename))
    return poses


ARCore_poses = get_poses_from_dir("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/threejs_data_exported/arcore_poses")
# COLMAP_poses = get_poses_from_dir("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/global_poses")
# Relative_poses = get_poses_from_dir("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/relative_poses")


def draw_poses(poses, r, g, b):
    for pose in poses:
        bpy.ops.mesh.primitive_cone_add(radius1=0, radius2=1, depth=2, enter_editmode=False, location=(0, 0, 0))
        cone = bpy.context.active_object
        # bpy.context.object.rotation_euler[0] = 3.1415926
        # bpy.context.object.location[2] = 1
        # bpy.ops.mesh.primitive_cube_add(enter_editmode=False, location=(0, 0, 0))
        # cube = bpy.context.active_object
        # cube.select = True
        cone.select = True
        # bpy.ops.object.join()

        # the cone and cube is called "cam"
        cam = bpy.context.active_object

        # set color
        mat = bpy.data.materials.new("PKHG")
        mat.diffuse_color = (r, g, b)
        o = bpy.context.selected_objects[0]
        o.active_material = mat

        bpy.ops.object.select_all(action='DESELECT')
        cam.select = True

        pose = mathutils.Matrix(pose)
        rotm = pose.to_3x3().to_4x4()
        cam.matrix_world = rotm

        tvec = pose.to_translation()
        camera_center = -np.linalg.inv(rotm.to_3x3()).dot(tvec)

        # rotm = pose[0:3, 0:3]
        # tvec = pose[0:3,3]
        # rotm = mathutils.Matrix(rotm)
        # euler = rotm.to_euler("XYZ")
        # camera_center = -np.linalg.inv(rotm).dot(tvec)

        # cam.rotation_euler.x = euler[0]
        # cam.rotation_euler.y = euler[1]
        # cam.rotation_euler.z = euler[2]

        cam.location.x = tvec[0]
        cam.location.y = tvec[1]
        cam.location.z = tvec[2]

        cam.scale = (0.04, 0.04, 0.04)


draw_poses(ARCore_poses, 1, 0, 0)
# draw_poses(Relative_poses, 0, 1 ,0)