

import sys
import os
import pandas as pd
import json
import gzip
import cv2
import numpy as np

# from scipy.spatial import cKDTree
# import open3d as o3d

# def chamfer_distance():
#     # calculation method taken from https://github.com/UP-RS-ESP/PointCloudWorkshop-May2022/blob/main/2_Alignment/Distance_metrics_between_pointclouds.ipynb

#     A = np.asarray(pcd_p.points)
#     B = np.asarray(pcd_q.points)
#     d1, _ = cKDTree(A).query(B, k=1, workers=-1)
#     d2, _ = cKDTree(B).query(A, k=1, workers=-1)

#     chamfer_d = np.mean(np.square(d1)) + np.mean(np.square(d2))
#     return chamfer_d

# will do at end. Need to compare with nerfacto, some how

def min_max_normalize(depth_array):
    return (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))

def depth_error(gt_depth_png_file, est_depth_npy_gz_file):
    # ground truth depth
    depth_array_gt = cv2.imread(gt_depth_png_file, cv2.IMREAD_GRAYSCALE)
    depth_array_gt = min_max_normalize(depth_array_gt)
    
    # estimated depth
    with gzip.open(est_depth_npy_gz_file, 'rb') as f:
        depth_array_est = 1/np.load(f)
        depth_array_est = min_max_normalize(depth_array_est)

    l2_norm = np.linalg.norm(depth_array_gt - depth_array_est)
    return l2_norm

def get_depth_files(input_folder, render_folder):
    [dataset_type, prefix, method_name, dataset_name, date, time] = render_folder.split("_")
    gt_depth_png_file = ""
    
    if dataset_type == "synthetic":
        if dataset_name == "hotdog":
            gt_depth_png_file = "data/blender/hotdog/test/r_1_depth.png"
        else:
            gt_depth_png_file = f"data/blender/{dataset_name}/r_1_depth.png"

    est_depth_npy_gz_file = f"{input_folder}{render_folder}/test/raw-depth/r_1.npy.gz"

    return gt_depth_png_file, est_depth_npy_gz_file

def jsons_to_excel(input_folder, output_excel):
    # Get a list of JSON files in the input folder
    json_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.json')]
    
    df = pd.DataFrame()
    df["Metrics"] = ["psnr", "ssim", "lpips", "num_rays_per_sec", "depth_err_m"]

    for json_file in json_files:
        with open(json_file) as f:
            json_data = json.load(f)
        results = json_data["results"]

        if results:
            file_name = os.path.basename(json_file)
            render_folder = file_name.replace(".json", "")
            
            # get depth data and calculate depth errpr
            gt_depth_png_file, est_depth_npy_gz_file = get_depth_files(input_folder, render_folder)
            depth_err = depth_error(gt_depth_png_file, est_depth_npy_gz_file)

            # point cloud file
            pt_cloud_ply = f"{input_folder}{render_folder}/point_cloud.ply"
            print("point_cloud_file --> ", pt_cloud_ply)

            df[file_name] = [results["psnr"], results["ssim"], results["lpips"], results["num_rays_per_sec"], depth_err]

    df.to_excel(output_excel, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: get_eval_results.py <input_folder> <output_excel>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_excel = sys.argv[2]
    jsons_to_excel(input_folder, output_excel)
