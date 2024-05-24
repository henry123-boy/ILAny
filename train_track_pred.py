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
from single_script import DiT_models
from infer_track_pred import vis_tracks

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from dataloader import TrackDataset


## GLOBAL variable
ACTION_DIM = 800 ## the max points are ACTION_DIM/2

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def find_model(model_name):
    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint



@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################



def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."


    # Setup DDP:
    dist.init_process_group("nccl")
    #dist.init_process_group(backend='nccl', init_method='env://', rank = torch.cuda.device_count(), world_size = 1)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        
        exp_details = 'trackexp' 
        
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}--{exp_details}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:

    model = DiT_models[args.model](
        num_points=args.num_points
    )

    if args.resume:

        ckpt_path = args.ckpt 
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict)
        logger.info(f"resuming from ckpt {ckpt_path}")


    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank] , find_unused_parameters=True)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    #opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    ############################## Setup data BEGIN ----------------------------


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

    ############################## Setup data END ----------------------------

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} examples")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        #sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for nbatch in loader:
            
            tar_img = nbatch['img_tar'].to(device).clone()/127.5 - 1.0    
            tar_tracks = nbatch['tracks_tar'].to(device).clone()
            tar_vis = nbatch['vis_tar'].to(device).clone()
            src_imgs = nbatch['imgs_src'].to(device).clone()/127.5 - 1.0   
            src_tracks = nbatch['tracks_src'].to(device).clone()

            x = tar_tracks.to(device).clone()
            ## choose a random number of points in a batch
            num_points = random.randint(100, ACTION_DIM) ##change as needed
            
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

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            model_kwargs = dict(src_imgs=src_imgs,
                                 src_tracks=src_tracks, 
                                 tar_img=tar_img, point_cond=tar_tracks[:,:1,...])
            # model_kwargs = dict(
            #     y=torch.cat([src_imgs[:,-1:], tar_img[:,None]], dim=1),
            # )

            if args.point_conditioned:
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs, noise=noise, point_conditioned=True)
            else:
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

            loss = loss_dict["loss"].mean()
            # print("t:", t, "loss:", loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)
            
            # Log loss values:
            running_loss += loss.detach().clone().item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                if train_steps % (20*args.log_every) == 0:
                    with torch.no_grad():
                        # visualization the results
                        naction = diffusion.p_sample_loop(model.forward, x.shape, noise, 
                                    clip_denoised=False, model_kwargs=model_kwargs,
                                    progress=True, device=device,point_conditioned=True,img=x)
                        pred_track = naction.view(x.shape[0],16,2,num_points).permute(0,3,1,2)
                        gt_track = x.view(x.shape[0],16,2,num_points).permute(0,3,1,2)
                        
                        # vis the pred track
                        from moviepy.editor import ImageSequenceClip
                        tar_img_vis = torch.nn.functional.interpolate(tar_img, (256,256))
                        action_pred = (pred_track.clone().detach().cpu().numpy() + 1)/2
                        action_pred[...,0] *= 256
                        action_pred[...,1] *= 256
                        new_frame_size = (256,256)
                        res_pred_video = vis_tracks(tar_img_vis, action_pred[0].transpose(1,0,2), action_pred.shape[2], new_frame_size)
                        video_pred_out = torch.from_numpy(np.stack(res_pred_video)).permute(0, 3, 1, 2)[None].byte()
                        # cv2.imwrite("pred_track.png", video_pred_out[0,0].permute(1, 2, 0).cpu().numpy())
                        wide_pred_list = list(video_pred_out.unbind(1))
                        wide_pred_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_pred_list]
                        clip = ImageSequenceClip(wide_pred_list, fps=1)
                        clip.write_gif(os.path.join("./", f"pred_track.gif"))
                        # vis the gt track
                        action = (gt_track.clone().detach().cpu().numpy() + 1)/2
                        action[...,0] *= 256
                        action[...,1] *= 256
                        new_frame_size = (256,256)
                        res_gt_video = vis_tracks(tar_img_vis, action[0].transpose(1,0,2), action.shape[2], new_frame_size)
                        video_gt_out = torch.from_numpy(np.stack(res_gt_video)).permute(0, 3, 1, 2)[None].byte()
                        # cv2.imwrite("gt_track.png", video_gt_out[0,0].permute(1, 2, 0).cpu().numpy())
                        wide_list = list(video_gt_out.unbind(1))
                        wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
                        clip = ImageSequenceClip(wide_list, fps=1)
                        clip.write_gif(os.path.join("./", f"gt_track.gif"))
                        print("the final track loss:", torch.mean((pred_track - gt_track).norm(dim=-1)).item())
                
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.detach().clone().item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="./checkpoints/homanga/TEST_FOLDER")
    parser.add_argument("--model", type=str,  default="DiT-L/2-NoPosEmb")
    parser.add_argument("--num_points", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=6000)
    parser.add_argument("--point_conditioned", type=bool, default=True)
    parser.add_argument("--resume", type=bool, default=False,help="whether to resume from a ckpt")
    parser.add_argument("--ckpt", type=str, help="Optional path to a DiT checkpoint  to resume trainining is needed")


    args = parser.parse_args()
    main(args)


