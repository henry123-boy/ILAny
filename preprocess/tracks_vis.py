
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
import decord

# import the tracking utils
from preprocess import (
    sample_from_mask,
    sample_double_grid,
    track_and_remove,
    Visualizer,
)


def extract_tracks(rgb, track,
                     CUDA_ID, idx=32):
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

    cotracker = torch.hub.load(os.path.join(os.path.expanduser("~"),
                                             "/nas3/xyx/cache_backup/facebookresearch_co-tracker_main/"),
                                             "cotracker_w8", source="local")
    cotracker = cotracker.eval().to(device)

    video_path = os.path.join(".", 'videos')
    if not os.path.exists(video_path):
        os.makedirs(video_path, exist_ok=True)

    num_points = 1000
    num_grid_points = 32
    add_mode = False

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
            pred_grid_tracks = torch.zeros((1, T, num_grid_points, 2)).to(device)
            pred_grid_vis = torch.ones((1, T, num_grid_points)).to(device)
        
        velocity = (pred_tracks[:, 1:] - 
                    pred_tracks[:, :-1]).norm(dim=-1).mean(dim=-1)  
        idx = torch.sort(torch.sort(velocity, descending=True).indices[:,:idx]).values[0]
        pred_tracks = pred_tracks[:, idx]
        pred_vis = pred_vis[:, idx]
        rgb = rgb[idx]
        # vis.visualize(rgb[None], pred_tracks, pred_vis, filename=f"test")
    
    # clear the cache
    torch.cuda.empty_cache()
    return rgb, pred_tracks, pred_vis

SRC_DIR = "/nas3/xyx/RH20T/RH20T_cfg2/task_0051_user_0007_scene_0001_cfg_0002_human/cam_104122063678/color.mp4"
TAR_DIR = "/nas3/xyx/RH20T/RH20T_cfg2/task_0051_user_0007_scene_0001_cfg_0002/cam_104122063678/color.mp4"

vr_src = decord.VideoReader(SRC_DIR)
vr_tar = decord.VideoReader(TAR_DIR)
sample_index = list(range(0, len(vr_src), 1))
sample_index_tar = list(range(0, len(vr_tar), len(vr_tar)//100))
video_src = vr_src.get_batch(sample_index).asnumpy()
video_tar = vr_tar.get_batch(sample_index_tar).asnumpy()

vis_ = Visualizer(save_dir="./videos", pad_value=0, fps=7, tracks_leave_trace=3)
rgbs_src, tracks_src, vis_src = extract_tracks(video_src, track=True, CUDA_ID=0, idx=32)
vis_.visualize(rgbs_src[None], tracks_src, vis_src, filename="src")

vis_future = Visualizer(save_dir="./videos", pad_value=0, fps=7, tracks_leave_trace=16, mode="cool", linewidth=1)
rgbs_tar, tracks_tar, vis_tar = extract_tracks(video_tar, track=True, CUDA_ID=0, idx=100)
pick_idx = torch.randperm(tracks_tar.shape[2])

from matplotlib import cm
import matplotlib.pyplot as plt
color_map = cm.get_cmap("gist_rainbow")
T = ACTION_HORIZON = 16
N = 800
vector_colors = np.zeros((T, N, 3))
linewidth = 1

## visualize tracks 
res_video = []
rand_idx = np.random.randint(len(rgbs_tar)-16)
rgb = (rgbs_tar[rand_idx]).permute(1,2,0).byte().detach().cpu().numpy() ## initial image
tracks = tracks_tar[0, rand_idx:rand_idx+16].detach().cpu().numpy()
H,W,C = rgb.shape

### RESHAPE image 
# rgb = cv2.resize(rgb, new_frame_size, interpolation=cv2.INTER_CUBIC)

_, N, D = tracks.shape

H,W,C = rgb.shape
assert C == 3

for i in range(ACTION_HORIZON):
    ## MAKE VIDEO BLACK / alpha less
    a_channel = np.ones(rgb.shape, dtype=np.float64)/2.0
    res_video.append(rgb*a_channel)

T = ACTION_HORIZON

color_map = cm.get_cmap("gist_rainbow")
vector_colors = np.zeros((T, N, 3))
linewidth = 1
segm_mask = None
query_frame = 0

if segm_mask is None:
    y_min, y_max = (
        tracks[query_frame, :, 1].min(),
        tracks[query_frame, :, 1].max(),
    )
    norm = plt.Normalize(y_min, y_max)
    for n in range(N):
        color = color_map(norm(tracks[query_frame, n, 1]))
        color = np.array(color[:3])[None] * 255
        vector_colors[:, n] = np.repeat(color, T, axis=0)

else:
    vector_colors[:, segm_mask <= 0, :] = 255

    y_min, y_max = (
        tracks[0, segm_mask > 0, 1].min(),
        tracks[0, segm_mask > 0, 1].max(),
    )
    norm = plt.Normalize(y_min, y_max)
    for n in range(N):
        if segm_mask[n] > 0:
            color = color_map(norm(tracks[0, n, 1]))
            color = np.array(color[:3])[None] * 255
            vector_colors[:, n] = np.repeat(color, T, axis=0) 

for t in range(T):
    if t >0:
        res_video[t] = res_video[t-1].copy()
    for i in range(N):
        coord = (int(tracks[t, i, 0]), int(tracks[t, i, 1]))

        if t > 0:
            coord_1 = (int(tracks[t-1, i, 0]), int(tracks[t-1, i, 1]))


        if coord[0] != 0 and coord[1] != 0:
                
                cv2.circle(res_video[t],coord,int(linewidth * 1),vector_colors[t, i].tolist(),thickness=-1 -1,)

                if t > 0:
                    cv2.line(res_video[t],coord_1,coord,vector_colors[t, i].tolist(),thickness=int(linewidth * 1))

video_out = torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()
#print("video",video.shape)
save_dir = 'save_tracK_pred/'
filename = 'test_vis'
os.makedirs(save_dir, exist_ok=True)
wide_list = list(video_out.unbind(1))
wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]

from moviepy.editor import ImageSequenceClip
clip = ImageSequenceClip(wide_list, fps=2)

# Write the video file
save_path = os.path.join(save_dir, f"{filename}_pred_track.mp4")
clip.write_videofile(save_path, codec="libx264", fps=2, logger=None)

## save a GIF too
from moviepy.editor import VideoFileClip
            
# loading video dsa gfg intro video
clip = VideoFileClip(os.path.join(save_dir, f"{filename}_pred_track.mp4"))

# saving video clip as gif
clip.write_gif(os.path.join(save_dir, f"{filename}_pred_track_out.gif"))
print(f"Video saved to {save_path}")        


# vis_future.visualize(rgbs_tar[None], tracks_tar[:,:,pick_idx], vis_tar[:,:,pick_idx], filename="tar", future=True)

import ipdb; ipdb.set_trace()
