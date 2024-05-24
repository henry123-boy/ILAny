"""
Extract tracks from the dataset

1. Using the Cotracker: https://github.com/facebookresearch/co-tracker to extract 
those 2d tracks

2. We also borrow from ATM for data preprocessing

#NOTE:
SOURCE DATA_STRUCTURE:
    ROOT_DIR
    ├── subset_xxx
        ├── color

    |—— subset_xxx
        |—— the same structure 
    | .....
    |—— task_description.json

#NOTE:
#TODO: define the structure of the destination data
DESTINATION DATA_STRUCTURE:

    DEST_DIR
    ├── 000
        ├── hash_num.hdf5
            cam_x: ...

    |—— 001
        |—— the same structure 
    | .....
    |—— task_mapping.npy        # the language mapping for each folder

"""

import os, sys
import argparse
import json
import click
import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange
import cv2
from multiprocessing import Pool
import concurrent.futures
import h5py
import time
import torch.nn.functional as F
# from pcache_fileio import fileio
import decord
import argparse
import glob

# import the tracking utils
from preprocess import (
    sample_from_mask,
    sample_double_grid,
    track_and_remove,
    Visualizer
)

def hash_map(info: list):

    """
    Hash the teleoperation information into hash keys.
    For a instance: 
    user: 0007 scene: 0005 cfg: 0002 
    
    The hash key is the hash of the string "200050007"

    """
    user_num = info[1] 
    scene_num = info[3]
    cfg_num = info[5]
    if len(info) == 7:
        human_num = "100"
    else:
        human_num = "000"
    key = f"{int(cfg_num)}{scene_num}{user_num}{human_num}"

    return key

def extract_tracks(rgb, track,
                    cam_name, f, CUDA_ID, RGB_DIR):
    """
    Extract the tracks from the given rgb images, and select those keyframes by
    the changes speed of tracks.
    
    Args:
        - rgb: the rgb images;  
        - track: whether to track the points.
        - cam_name: `str` the name of camera
        - f: this is the handler for the hdf5 file 
    """

    # set CUDA
    device = torch.device(f"cuda:{CUDA_ID}")
    # create group 
    f.update({cam_name: {}})

    cotracker = torch.hub.load("/nas3/xyx/cache_backup/facebookresearch_co-tracker_main/", "cotracker_w8", source="local")
    cotracker = cotracker.eval().to(device)
    video_path = os.path.join(".", 'videos')
    if not os.path.exists(video_path):
        os.makedirs(video_path, exist_ok=True)

    num_points = 1000

    with torch.no_grad():
        
        T, H, W, C = rgb.shape
        rgb = rearrange(torch.from_numpy(rgb).float(), 't h w c -> t c h w').to(device)
        # sample random points
        points = sample_from_mask(np.ones((H, W, 1)) * 255, num_samples=num_points)
        points = torch.from_numpy(points).float().to(device)
        points = torch.cat(
            [torch.ones_like(points[:, :1]) * torch.randint_like(points[:, :1], 0, T), points],
            dim=-1).to(device)

        # sample grid points
        grid_points = sample_double_grid(7, device=device)
        grid_points[:, 0] = grid_points[:, 0] * H
        grid_points[:, 1] = grid_points[:, 1] * W
        grid_points = torch.cat(
            [torch.ones_like(grid_points[:, :1]) * torch.randint_like(grid_points[:, :1], 0, T),
                grid_points], dim=-1).to(device)

        if track:
            pred_tracks, pred_vis = track_and_remove(cotracker, rgb[None], points[None])
            # pred_grid_tracks, pred_grid_vis = track_and_remove(cotracker, rgb[None], grid_points[None], var_threshold=0.)
        else:
            pred_tracks = torch.zeros((1, T, num_points, 2)).to(device)
            pred_vis = torch.ones((1, T, num_points)).to(device)
        
        velocity = (pred_tracks[:, 1:] - 
                    pred_tracks[:, :-1]).norm(dim=-1).mean(dim=-1)        
        idx = torch.where(velocity > 0.5)[1]
        # pred_tracks = pred_tracks[:, idx]
        # pred_vis = pred_vis[:, idx]
        # vis.visualize(rgb[None], pred_tracks, pred_vis, filename=f"test")
    
    pred_tracks[:, :, :, 0] /= W
    pred_tracks[:, :, :, 1] /= H
    
    f[cam_name].update({"tracks": pred_tracks[0].cpu().numpy()})
    f[cam_name].update({"vis": pred_vis[0].cpu().numpy()})
    f[cam_name].update({"idx": idx.cpu().numpy()})
    f[cam_name].update({"mp4_dir": RGB_DIR})
    # clear the cache
    torch.cuda.empty_cache()
    return True

def submit_subtask(task_traj_i, 
                   DEST_DIR, SUBSET_ROOT, num_workers_cam=8, GPU_NUM=3):
    
    Task_i = task_traj_i.split('_')[0] + "_" + task_traj_i.split('_')[1]
    # locate its corresponding task
    TASK_ROOT_I = f"{DEST_DIR}/{Task_i}"
    # the reset of information
    Reset_info_i = task_traj_i.split('_')[2:]
    hash_key = hash_map(Reset_info_i)
    # check if the file exists
    if os.path.exists(f"{TASK_ROOT_I}/{hash_key}.npy"):
        print(f"Task: {task_traj_i} has been processed")
        return
    # create the file stream
    f_dict = dict({})

    # extract the tracks from multiview camera
    TRAJ_DIR = f"{SUBSET_ROOT}/{task_traj_i}"

    CAMS = os.listdir(TRAJ_DIR)
    CAMS = [i for i in CAMS if "cam_" in i]
    CAMS_GROUP = [CAMS[i*num_workers_cam:(i+1)*num_workers_cam] for i in range(len(CAMS) // num_workers_cam + 1)]
    # multiprocessing the tracking
    occ_fold_list = []
    for CAM_GROUP_I in CAMS_GROUP:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for j, CAM_I in enumerate(CAM_GROUP_I):
                CUDA_ID = j%GPU_NUM
                RGB_DIR = f"{TRAJ_DIR}/{CAM_I}/color.mp4" 
                # tell this vide is occupied by some threads
                occ_fold = f"{TRAJ_DIR}/{CAM_I}/occupy"  
                # load the image
                if os.path.exists(occ_fold)==True:
                    continue
                # make the occupied folder
                os.makedirs(occ_fold, exist_ok=True)
                occ_fold_list.append(occ_fold)
                if os.path.exists(RGB_DIR)==False:
                    continue
                with open(RGB_DIR, 'rb') as f:
                    vr = decord.VideoReader(RGB_DIR)
                sample_index = list(range(0, len(vr), 1))
                imgs = vr.get_batch(sample_index).asnumpy()
                print(f"submit the task: {task_traj_i} and camera: {CAM_I}")
                executor.submit(extract_tracks,
                                imgs, True, CAM_I, f_dict, CUDA_ID, RGB_DIR)

            executor.shutdown(wait=True)
    # close the file stream
    np.save(f"{TASK_ROOT_I}/{hash_key}.npy", f_dict)
    for occ_fold_i in occ_fold_list:
        os.rmdir(occ_fold_i)

parser = argparse.ArgumentParser(description="Extract the tracks from the dataset")
parser.add_argument("--root_dir", type=str,
                     default="/nas3/xyx/RH20T", help="The root directory of the dataset")
parser.add_argument("--dest_dir", type=str,
                     default="/nas3/xyx/ILAnyDATA", help="The destination directory of the dataset")
parser.add_argument("--gpu_num", type=int,
                     default=3, help="The number of GPU")
args = parser.parse_args()

if __name__ == "__main__":
    
    ROOT_DIR = args.root_dir
    # occ_fold_list = glob.glob(f"{ROOT_DIR}/*/*/*/occupy")
    # for occ_fold_i in occ_fold_list:
    #     os.rmdir(occ_fold_i)
    # import ipdb; ipdb.set_trace()
    
    DEST_DIR = args.dest_dir
    os.makedirs(DEST_DIR, exist_ok=True)
    # load the task description
    JSON_DIR = f"{ROOT_DIR}/task_description.json"
    with open(JSON_DIR, 'r') as f:
        task_desc = json.load(f)
    # those subset of data
    SUBSET = os.listdir(ROOT_DIR)
    SUBSET = [folder for folder in SUBSET if os.path.isdir(f"{ROOT_DIR}/{folder}")]
    # the number of GPU
    GPU_NUM = int(args.gpu_num)
    # assert GPU_NUM % 2 ==0 , "The number of GPU should be divided by 2"
    
    num_workers_all = GPU_NUM * 6 # set the number of workers as the 12 times of gpu
    num_workers_task = num_workers_all // 8 # the number of workers for different task
    num_workers_cam = 8 # the number of workers for different camera

    tasknames = list(task_desc.keys())

    for subset_i in SUBSET:
        SUBSET_ROOT = f"{ROOT_DIR}/{subset_i}"
        # extract the different tasks
        TASKS_Traj = os.listdir(SUBSET_ROOT)[2000:]

        # create the tasks folder
        for task_i in tasknames:
            task_root = f"{DEST_DIR}/{task_i}"
            os.makedirs(task_root, exist_ok=True)

        TASKS_GROUP = [TASKS_Traj[i*num_workers_task:(i+1)*num_workers_task] 
                       for i in range(len(TASKS_Traj) // num_workers_task + 1)]

        for TASKS_GROUP_I in tqdm(TASKS_GROUP):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i, TASKS_GROUP_II in enumerate(TASKS_GROUP_I):
                    executor.submit(submit_subtask, TASKS_GROUP_II, DEST_DIR, SUBSET_ROOT, 8, GPU_NUM)
                executor.shutdown(wait=True)
                


        



