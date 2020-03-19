import time
import numpy as np
import struct
import collections

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

# TODO: replace this method with the text version and get the last one ?
def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)

    return images

def get_query_image_global_pose(name):
    images = read_images_binary("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/model/0/images.bin")
    for k,v in images.items():
        if v.name == name:
            image = v
    pose_r = image.qvec2rotmat()
    pose_t = image.tvec
    pose = np.c_[pose_r, pose_t]
    pose = np.r_[pose, [np.array([0, 0, 0, 1])]]
    return pose

"""
The reconstructed pose of an image is specified as the projection 
from world to the camera coordinate system of an image using a quaternion (QW, QX, QY, QZ) 
and a translation vector (TX, TY, TZ). 
"""
def get_query_image_global_pose_new_model(name):
    images = read_images_binary("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/new_model/images.bin")
    for k,v in images.items():
        if v.name == name:
            image = v
    pose_r = image.qvec2rotmat()
    pose_t = image.tvec
    pose = np.c_[pose_r, pose_t]
    pose = np.r_[pose, [np.array([0, 0, 0, 1])]]
    return pose

def get_query_image_global_pose_new_model_quaternion(name):
    images = read_images_binary("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/new_model/images.bin")
    for k,v in images.items():
        if v.name == name:
            image = v
    pose_r = image.qvec
    pose_t = image.tvec
    pose = np.r_[pose_t, [pose_r[1],pose_r[2],pose_r[3]],pose_r[0]]
    return pose

def get_query_image_id(name):
    images = read_images_binary("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/model/0/images.bin")
    for k,v in images.items():
        if v.name == name:
            image = v
    id = image.id
    return id

def get_query_image_id_new_model(name):
    images = read_images_binary("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/new_model/images.bin")
    for k,v in images.items():
        if v.name == name:
            image = v
    id = image.id
    return id