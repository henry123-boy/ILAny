# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import random 
import cv2 
import cv2
import skimage.transform as st
from skvideo.io import vwrite
import os
import torch.nn as nn
import torchvision
import collections
import decord

import pickle

from torch.nn import functional as F
from torchvision.datasets.utils import download_url

import imageio

from scipy.spatial.transform import Rotation 
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d


from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import random 
import cv2 
import cv2
import skimage.transform as st
from skvideo.io import vwrite
import os
import torch.nn as nn
import torchvision
import collections
import torch.nn.functional as F
# from models import DiT_models
from single_script import DiT_models as DiT_models_track

from matplotlib import cm
import matplotlib.pyplot as plt
from dataloader import TrackDataset


def find_model(model_name):
    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint




def read_video_from_path(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video file")
    else:
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.resize(frame,(128,128))
                frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                break
        cap.release()

    return np.stack(frames)

def get_filename_without_extension(path):
    return os.path.splitext(os.path.basename(path))[0]



# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    for i in range(len(stats['min'])):
        stats['min'][i] = 0
        stats['max'][i] = 96

    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    for i in range(len(stats['min'])):
        stats['min'][i] = 0
        stats['max'][i] = 96
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    for i in range(len(stats['min'])):
        stats['min'][i] = 0
        stats['max'][i] = 96

    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


## Choose points in the mask to condition prediction on 
def meshgrid2d(B, Y, X, stack=False, norm=False, device="cuda"):
    # returns a meshgrid sized B x Y x X

    grid_y = torch.linspace(0.0, Y - 1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X - 1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if stack:
        # note we stack in xy order
        # (see https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x

def get_points_on_a_grid(grid_size, interp_shape, grid_center=(0, 0), device="cuda"):
    if grid_size == 1:
        return torch.tensor([interp_shape[1] / 2, interp_shape[0] / 2], device=device)[
            None, None
        ]

    grid_y, grid_x = meshgrid2d(
        1, grid_size, grid_size, stack=False, norm=False, device=device
    )
    step = interp_shape[1] // 64
    if grid_center[0] != 0 or grid_center[1] != 0:
        grid_y = grid_y - grid_size / 2.0
        grid_x = grid_x - grid_size / 2.0
    grid_y = step + grid_y.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[0] - step * 2
    )
    grid_x = step + grid_x.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[1] - step * 2
    )

    grid_y = grid_y + grid_center[0]
    grid_x = grid_x + grid_center[1]
    xy = torch.stack([grid_x, grid_y], dim=-1).to(device)
    return xy


## GLOBAL variable
ACTION_DIM = 800 ## 2* number of points
ACTION_HORIZON = 16 ## pred horizon

def main(args):
    device = "cuda:0"
    # parameters
    pred_horizon = 8#8#16 ## how many time-steps for prediction ## needs to be a power of 2
    obs_horizon = 2 ## how many images to condition on
    action_horizon = 8 ## IGNORE
    # create dataset from file
    dataset = TrackDataset(
        data_path=args.data_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    eval_data = "from dataset"
    if eval_data == "from dataset":
        for i in range(2):
             data_from_loader = next(iter(loader))
        tar_img = (data_from_loader['img_tar']/127.5 - 1.0).to(device)
        tar_tracks = data_from_loader['tracks_tar'].to(device)
        src_imgs = (data_from_loader['imgs_src']/127.5 - 1.0).to(device)
    else:    
        shp_default = (140, 140)
        tar_msk = (cv2.imread("./assets/target_img_msk0.png", cv2.IMREAD_GRAYSCALE)>-100)
        tar_msk = cv2.resize(tar_msk.astype(np.uint8), (140, 140))
        tar_img = cv2.imread("./assets/target_img2.png")   
        tar_img = cv2.resize(tar_img, (140, 140))
        tar_img = torch.from_numpy(tar_img.transpose(2, 0, 1)).float().to(device) / 127.5 - 1.0
        tar_img = tar_img[None]
        tar_tracks = get_points_on_a_grid(10, shp_default, grid_center=(0, 0), device=device)
        point_mask = tar_msk[
                    (tar_tracks[0, :, 1]).round().long().cpu(),
                    (tar_tracks[0, :, 0]).round().long().cpu(),
                ]
        tar_tracks = tar_tracks[:, point_mask]
        tar_tracks[...,0] /= shp_default[1]
        tar_tracks[...,1] /= shp_default[0]
        tar_tracks = tar_tracks[None].repeat(1, 16, 1, 1)
        # cv2.imwrite("./assets/target_img_res.png", tar_img)
        vr = decord.VideoReader("./assets/src_video0.mp4")
        src_imgs = vr.get_batch(np.arange(0, len(vr), len(vr)//32)).asnumpy()[:32]
        src_imgs = torch.from_numpy(src_imgs.transpose(0, 3, 1, 2)).float().to(device) / 127.5 - 1.0
        src_imgs = torch.nn.functional.interpolate(src_imgs, size=shp_default, mode="bilinear", align_corners=False)[None]

    x = tar_tracks.to(device).clone()
    
    ## choose a random number of points in a batch
    num_points = tar_tracks.shape[2] ##change as needed
    
    if num_points%2 != 0: 
        num_points = num_points -1
    x = x[:,:,:num_points].clone()

    noise = torch.randn_like(x)
    # noise[:,0,:] = 0*noise[:,0,:] ## do not add noise to the first step

    x = x.permute(0,2,1,3)
    x = torch.reshape(x,(x.shape[0],num_points,16*2)).clone()
    x = x.permute(0,2,1).clone()

    noise = noise.permute(0,2,1,3)
    noise = torch.reshape(noise,(noise.shape[0],num_points,16*2)).clone()
    noise = noise.permute(0,2,1).clone()

    model_kwargs = dict(src_imgs=src_imgs,
                            src_tracks=None, 
                            tar_img=tar_img, point_cond=tar_tracks[:,:1,...])
    y = tar_img
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    if args.ckpt is None:
        assert 1==2

    # Load model:
    model = DiT_models_track[args.model](
        num_points=args.num_points
    ).to(device)

    ckpt_path = args.ckpt 
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))

    naction = diffusion.p_sample_loop(model.forward, x.shape, noise, 
                                    clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device,point_conditioned=True,img=x)
    

    # t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
    # diffusion.training_losses(model, x, t, model_kwargs, noise=noise, point_conditioned=True)

    #####
    naction = (naction.permute(0,2,1) + 1) / 2
    naction = naction.reshape(1,num_points,-1,2)
    action = naction[0].permute(1,0,2).clone().detach().cpu().numpy()


    ## scale point predictions to original 640x480 dim size of video
    new_frame_size = (256, 256) #(640, 480)

    # Calculate scaling factors
    sx = new_frame_size[0] / tar_img.shape[-1]
    sy = new_frame_size[1] / tar_img.shape[-2]

    # Scale the point locations to correspond to the new frame size
    scaled_action = action.copy()
    # scaled_action[..., 0] *= sx  # Scale x coordinates
    # scaled_action[..., 1] *= sy  # Scale y coordinates
    scaled_action[..., 0] *= 256  # Scale x coordinates
    scaled_action[..., 1] *= 256

    tar_tracks = (tar_tracks + 1) / 2 
    tar_tracks[...,0] *= 256
    tar_tracks[...,1] *= 256

    action = scaled_action

    
    if args.visualize:
        from matplotlib import cm
        color_map = cm.get_cmap("gist_rainbow")
        T = 16
        N = num_points
        vector_colors = np.zeros((T, N, 3))
        linewidth = 1

        ## visualize tracks 
        tracks = action
        res_video = vis_tracks(y, action, ACTION_HORIZON, new_frame_size)
        res_video_gt = vis_tracks(y, tar_tracks[0].detach().cpu().numpy(), ACTION_HORIZON, new_frame_size)

        video_out = torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()
        video_out_gt = torch.from_numpy(np.stack(res_video_gt)).permute(0, 3, 1, 2)[None].byte()
        #print("video",video.shape)
        save_dir = 'save_tracK_pred/'
        filename = 'test_vis'
        os.makedirs(save_dir, exist_ok=True)
        wide_list = list(video_out.unbind(1))
        wide_list_gt = list(video_out_gt.unbind(1))
        wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
        wide_list_gt = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list_gt]
        # visualize the source video
        src_img_list = ((src_imgs[0]+1)*127.5).detach().cpu().numpy().astype(np.uint8)
        src_img_list = [cv2.resize(src_img_list[i].transpose(1, 2, 0),
                                        new_frame_size, interpolation=cv2.INTER_CUBIC) for i in range(src_img_list.shape[0])]

        from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip(wide_list, fps=2)
        clip_gt = ImageSequenceClip(wide_list_gt, fps=2)
        clip_src = ImageSequenceClip(src_img_list, fps=2)
        clip_src.write_gif(os.path.join(save_dir, f"{filename}_src_video.gif"))
        clip_gt.write_gif(os.path.join(save_dir, f"{filename}_gt_track.gif"))
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

def vis_tracks(imgs, tracks, T, frame_size):
    res_video = []
    rgb = ((imgs[0]+1)*127.5).permute(1,2,0).byte().detach().cpu().numpy() ## initial image
    ### RESHAPE image 
    rgb = cv2.resize(rgb, frame_size, interpolation=cv2.INTER_CUBIC)

    _, N, D = tracks.shape

    H,W,C = rgb.shape
    assert C == 3

    for i in range(T):
        ## MAKE VIDEO BLACK / alpha less
        a_channel = np.ones(rgb.shape, dtype=np.float64)/2.0
        res_video.append(rgb*a_channel)

    T = T

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
    return res_video


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_points", type=int, default=25)
    parser.add_argument("--visualize", type=bool, default=True,help="whether to visualize predicted tracks")
    parser.add_argument("--model", type=str, choices=list(DiT_models_track.keys()), default="DiT-L/2-NoPosEmb")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--init", type=str, default='assets/init_test.jpg',
                        help="path to initial image")
    parser.add_argument("--goal", type=str, default='assets/goal_test.jpg',
                        help="path to goal image")
    parser.add_argument("--ckpt", type=str, default='/home/xyx/codes/ILAny/checkpoints/homanga/TEST_FOLDER/001-DiT-L-2-NoPosEmb--trackexp/checkpoints/0012000.pt',
                        help="path to trained checkpoint")

    args = parser.parse_args()
    main(args)
