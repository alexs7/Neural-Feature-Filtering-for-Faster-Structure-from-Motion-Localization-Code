from point3D_loader import read_points3d_binary, read_points3d_default

base_points = read_points3d_default("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/tmp/base/points3D.bin")
live_points = read_points3d_default("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/tmp/points3D.bin")

id = 0
for k,v in base_points.items():
    if(len(v.image_ids) < len(live_points[k].image_ids)):
        id +=1
        # breakpoint()

print(id)
