

import sys
import os
import pandas as pd
import json
import gzip
import cv2
import numpy as np
import re
import concurrent.futures
from numba import jit
from scipy.spatial import cKDTree
import open3d as o3d
import imageio

def min_max_normalize(depth_array):
    if np.max(depth_array) - np.min(depth_array) == 0:
        print("error. something went wrong with the depth generation.")
        print("depth_array: --> ", depth_array)
        print("max, min: --> ", np.max(depth_array), np.min(depth_array))
    
    return (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))

@jit(nopython=True)
def compute_l2_norm(depth_array_gt, depth_array_est):
    diff = depth_array_gt - depth_array_est
    return np.sqrt(np.sum(diff ** 2))

def test_depth_files(input_folder, render_folder):
    est_depth_folder = f"{input_folder}{render_folder}/test/raw-depth"
    for root, dirs, files in os.walk(est_depth_folder):
        for file_name in files:
            est_depth = os.path.join(root, file_name)

            with gzip.open(est_depth, 'rb') as f:
                depth_array_est = 1 / np.load(f)
                depth_array_est = min_max_normalize(depth_array_est)
    
def depth_error_calc(id_depth, depth_pngs):
    [est_depth, gt_depth] = depth_pngs
    
    # ground truth depth
    depth_array_gt = cv2.imread(gt_depth, cv2.IMREAD_GRAYSCALE)
    depth_array_gt = min_max_normalize(depth_array_gt)

    # estimated depth
    with gzip.open(est_depth, 'rb') as f:
        depth_array_est = 1 / np.load(f)
        depth_array_est = min_max_normalize(depth_array_est)
    norm = compute_l2_norm(depth_array_gt, depth_array_est) #np.linalg.norm(depth_array_gt - depth_array_est)
    return norm

def depth_error_all(depth_files):
    l2_norms = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(depth_error_calc, id_depth, depth_pngs): id_depth for id_depth, depth_pngs in depth_files.items()}
        for future in concurrent.futures.as_completed(futures):
            print("appending error: ", future.result())
            l2_norms.append(future.result())
    return sum(l2_norms)/len(l2_norms)

def get_depth_files(input_folder, render_folder):
    [dataset_type, prefix, method_name, dataset_name, date, time] = render_folder.split("_")
    depth_files = {} #key: id, value: [est depth, gt depth]
    gt_pattern = re.compile(r'r_(\d+)_depth\.png')
    est_pattern = re.compile(r'r_(\d+)\.npy\.gz')
    gt_depths = []

    # estimated depths
    est_depth_folder = f"{input_folder}{render_folder}/test/raw-depth"
    for root, dirs, files in os.walk(est_depth_folder):
        for file_name in files:
            id_depth = est_pattern.match(file_name).group(1)
            est_depth_path = os.path.join(root, file_name)
            depth_files[id_depth] = [est_depth_path]

    # ground truth depths
    gt_depth_folder = ""
    if dataset_type == "synthetic":
        gt_depth_folder= f"data/blender/{dataset_name}/test"
    
    for root, dirs, files in os.walk(gt_depth_folder):
        for file_name in files:
            gt_match = gt_pattern.match(file_name)
            if gt_match:
                id_depth = gt_match.group(1)
                if id_depth in depth_files:
                    gt_depth_path = os.path.join(root, file_name)
                    depth_files[id_depth].append(gt_depth_path)
                    gt_depths.append(gt_depth_path)
    
    return depth_files, gt_depths, dataset_name

def chamfer_distance(pt_cloud1_pts, pt_cloud2_pts):
    # calculation method taken from https://github.com/UP-RS-ESP/PointCloudWorkshop-May2022/blob/main/2_Alignment/Distance_metrics_between_pointclouds.ipynb

    A = np.asarray(pt_cloud1_pts)
    B = np.asarray(pt_cloud2_pts)
    d1, _ = cKDTree(A).query(B, k=1, workers=-1)
    d2, _ = cKDTree(B).query(A, k=1, workers=-1)

    chamfer_d = np.mean(np.square(d1)) + np.mean(np.square(d2))
    return chamfer_d

def compare_point_clouds(gt_depths, est_pt_cloud_ply, dataset_name):

    gt_filepath = f"data/blender/{dataset_name}/point_cloud_test.ply"
    
    if not os.path.exists(gt_filepath):
        num_points = 1000000
        comp_l = np.array([-1.5, -1.5, -1])  # bounding box min
        comp_m = np.array([1.5, 1.5, 2])  # bounding box max

        # get paramters
        meta_file = f"data/blender/{dataset_name}/transforms_test.json"
        with open(meta_file, 'r') as json_file:
            meta = json.load(json_file)
        
        first_img = f'data/blender/{dataset_name}/{meta["frames"][0]["file_path"].replace("./", "")}.png'
        img_0 = imageio.v2.imread(first_img)
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)
        fx = fy = focal_length
        cx = image_width / 2.0
        cy = image_height / 2.0

        # create ground truth point cloud
        pt_clouds = []
        for depth_file in gt_depths:
            depth_arr = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE)
            depth_arr = depth_arr.astype(np.float32) 
            depth_image = o3d.geometry.Image(depth_arr)
            selected_indices = np.random.choice(num_points, size=num_points, replace=False)

            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(image_width, image_height, fx, fy, cx, cy)

            point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth_image, intrinsic)
            pt_clouds.append(point_cloud.points)

        # assemble final pt cloud
        gt_point_cloud = o3d.geometry.PointCloud()
        gt_point_cloud.points = o3d.utility.Vector3dVector(np.vstack(pt_clouds)) 

        # remove outliers, apply to bounding box, and select max num_points
        pcd, ind = gt_point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=10)
        filtered_points_np = np.asarray(pcd.points)
        
        # scale v1
        # min_coords = np.min(filtered_points_np, axis=0)
        # max_coords = np.max(filtered_points_np, axis=0)
        # scale_factors = np.abs([1 / (max_coords[i] - min_coords[i]) for i in range(3)])
        # scaled_points = (filtered_points_np - min_coords) * scale_factors * 2 - 1
        
        # scale v2
        scaled_points = filtered_points_np/1e-3

        # mask = np.all((filtered_points_np > comp_l) & (filtered_points_np < comp_m), axis=-1)
        # filtered_points_masked = filtered_points_np[mask]

        # selected_indices_final = np.random.choice(len(scaled_points), size=num_points, replace=False)
        # filtered_points_final = scaled_points[selected_indices_final]
        
        gt_filtered = o3d.geometry.PointCloud()
        gt_filtered.points = o3d.utility.Vector3dVector(scaled_points)

        o3d.io.write_point_cloud(gt_filepath, gt_filtered)
        print(f"created point cloud for dataset {dataset_name}")
    else:
        gt_point_cloud = o3d.io.read_point_cloud(gt_filepath)

    # estimated point cloud
    est_point_cloud = o3d.io.read_point_cloud(est_pt_cloud_ply)

    print("Calculating Chamfer distance between point clouds ..")
    dist = chamfer_distance(gt_point_cloud.points, est_point_cloud.points)
    return dist

def jsons_to_excel(input_folder, output_excel):
    # Get a list of JSON files in the input folder
    json_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.json')]
    
    df = pd.DataFrame()
    df["Metrics"] = ["psnr", "ssim", "lpips", "num_rays_per_sec", "depth_avg_err_m", "chamfer_dist"]
    test_depth = False # test if depths generated correctly

    for json_file in json_files:
        with open(json_file) as f:
            json_data = json.load(f)
        results = json_data["results"]

        if results:
            file_name = os.path.basename(json_file)

            render_folder = file_name.replace(".json", "")

            print("-------- render: ", render_folder, " ------------- ")
            
            # get depth data and calculate depth errpr
            depth_err = 0
            if test_depth:
                test_depth_files(input_folder, render_folder)
            else:
                depth_files, gt_depths, dataset_name = get_depth_files(input_folder, render_folder)
                depth_err = depth_error_all(depth_files)

            # point cloud comparison
            chamfer_dist = 0
            # est_pt_cloud_ply = f"{input_folder}{render_folder}/point_cloud.ply"
            # if os.path.exists(est_pt_cloud_ply):
            #     chamfer_dist = compare_point_clouds(gt_depths, est_pt_cloud_ply, dataset_name)
            # else:
            #     print("Estimated point cloud was not created. ")


            df[file_name] = [results["psnr"], results["ssim"], results["lpips"], results["num_rays_per_sec"], depth_err, chamfer_dist]

    df.to_excel(output_excel, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: get_eval_results.py <input_folder> <output_excel>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_excel = sys.argv[2]
    jsons_to_excel(input_folder, output_excel)
