# Just printing stats for the models
import os
import sys
import numpy as np
from database import COLMAPDatabase
from query_image import get_all_images_names_from_db, get_localised_image_by_names

def print_stats_images(path, query_name_path=None):
    print("Images Stats:")
    if(query_name_path is None):
        base_db_path = os.path.join(path, 'database.db')
        base_db = COLMAPDatabase.connect(base_db_path)
        base_image_names = get_all_images_names_from_db(base_db)
        print(f"Total number of base images: {len(base_image_names)}")
    else:
        query_images_txt_path = os.path.join(path, "query_name.txt")
        live_image_names = np.loadtxt(query_images_txt_path, dtype=str) #only live images
        localised_live_images_names = get_localised_image_by_names(live_image_names, os.path.join(path, "model", "images.bin"))
        print(f"Total number of localised images: {len(localised_live_images_names)}")
    print()

# TODO Continue from here 28/02/2023 (2/2)

def print_stats_points(path, query_name_path=None): #Continue from here
    print("Images Stats:")
    if(query_name_path is None):
        base_db_path = os.path.join(path, 'database.db')
        base_db = COLMAPDatabase.connect(base_db_path)
        base_image_names = get_all_images_names_from_db(base_db)
        print(f"Total number of base images: {len(base_image_names)}")
    else:
        query_images_txt_path = os.path.join(path, "query_name.txt")
        live_image_names = np.loadtxt(query_images_txt_path, dtype=str) #only live images
        localised_live_images_names = get_localised_image_by_names(live_image_names, os.path.join(path, "model", "images.bin"))
        print(f"Total number of localised images: {len(localised_live_images_names)}")
    print()

dataset = sys.argv[1] #HGE, CAB, LIN (or Other for CMU, retail shop)

print("Stats for models")

if(dataset == "HGE" or dataset == "CAB" or dataset == "LIN"):
    print(dataset)
    path = f"/media/iNicosiaData/engd_data/lamar/{dataset}_colmap_model"
    path_mnm = f"/media/iNicosiaData/engd_data/lamar/{dataset}_colmap_model/exmaps_data/models_for_match_no_match"
    print("COLMAP SIFT models")
    print_stats_images(os.path.join(path, "base"))
    print_stats_images(os.path.join(path, "live"), True)
    print_stats_images(os.path.join(path, "gt"), True)
    print("OpenCV SIFT models")
    print_stats_images(os.path.join(path_mnm, "gt"), True)

if(dataset == "CMU"):
    if(len(sys.argv) > 2):
        slices_names = [sys.argv[2]]
    else: # do all slices
        slices_names = ["slice2", "slice3", "slice4", "slice5", "slice6", "slice7", "slice8", "slice9", "slice10", "slice11", "slice12", "slice13", "slice14", "slice15",
                        "slice16", "slice17", "slice18", "slice19", "slice20", "slice21", "slice22", "slice23", "slice24", "slice25"]
    for slice_name in slices_names:
        path = f"/media/iNicosiaData/engd_data/cmu/{slice_name}/exmaps_data"
        path_mnm = f"/media/iNicosiaData/engd_data/cmu/{slice_name}/exmaps_data/models_for_match_no_match"
        print(slice_name)
        print("COLMAP SIFT models")
        print_stats_images(os.path.join(path, "base"))
        print_stats_images(os.path.join(path, "live"), True)
        print_stats_images(os.path.join(path, "gt"), True)
        print("OpenCV SIFT models")
        print_stats_images(os.path.join(path_mnm, "gt"), True)

if(dataset == "RetailShop"):
    path = f"/media/iNicosiaData/engd_data/retail_shop/slice1/"
    path_mnm = f"/media/iNicosiaData/engd_data/retail_shop/slice1/models_for_match_no_match"
    print("COLMAP SIFT models")
    print_stats_images(os.path.join(path, "base"))
    print_stats_images(os.path.join(path, "live"), True)
    print_stats_images(os.path.join(path, "gt"), True)
    print("OpenCV SIFT models")
    print_stats_images(os.path.join(path_mnm, "gt"), True)
