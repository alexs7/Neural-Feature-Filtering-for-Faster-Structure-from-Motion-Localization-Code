from query_image import get_query_image_global_pose_new_model_quaternion
from query_image import get_query_image_global_pose_new_model

colmap_pose = get_query_image_global_pose_new_model_quaternion("query.jpg")
colmap_pose = str(colmap_pose.flatten().tolist())[1:-1]

print(colmap_pose)
