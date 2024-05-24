
import os,sys
import numpy as np
import cv2
import glob
import torch

from preprocess import (
    Visualizer
)
import decord

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    for i in range(len(stats['min'])): ## change depending on resolution
        stats['min'][i] = 0
        stats['max'][i] = 96

    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


# dataset
class TrackDataset(torch.utils.data.Dataset):
    def __init__(self,
                data_path: str,
                pred_horizon: int,
                obs_horizon: int,
                action_horizon: int):
        
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        if os.path.exists("./cleaned_data.npy"):
            samples = glob.glob(data_path + '*/*.npy')
            self.samples = list(np.load("./cleaned_data.npy"))
            self.samples = [dirs for dirs in self.samples if dirs in samples]
        else:
            samples = glob.glob(data_path + '*/*.npy')
            self.samples = [dirs for dirs in samples if 'task_emb_bert' not in dirs]

        self.samples = 5000*[self.samples[3]]



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        data_pick = self.samples[idx]
        PICK_DIFF_VIEW = None
        PICK_DIFF_CASE = None
        if np.random.rand()>0.5:
            PICK_DIFF_VIEW = True
        else:
            PICK_DIFF_VIEW = True
            PICK_DIFF_CASE = False

        if PICK_DIFF_VIEW:
            src_lang_dir = tar_lang_dir = data_pick.replace(data_pick.split("/")[-1],
                                                                        'task_emb_bert.npy')
            data_load = np.load(data_pick, allow_pickle=True).item()
            data_load.pop('cam_104422070042')
            cam_key = list(data_load.keys())
            cam_num = len(cam_key)
            if cam_num < 2:
                src_idx, ref_idx = 0, 0
            else:
                src_idx, ref_idx = torch.randperm(cam_num)[:2]
            # get the src and target camera info
            try:
                src = data_load[cam_key[src_idx]]
                target = data_load[cam_key[ref_idx]]
            except:
                print(len(cam_key), src_idx, ref_idx)
                os.remove(data_pick)
                raise ValueError

        if PICK_DIFF_CASE:
            task_name = data_pick.split('/')[-2]
            sub_samp = [dirs for dirs in self.samples if (task_name in dirs)&(dirs!=data_pick)]
            data_pick_ = sub_samp[np.random.randint(len(sub_samp))]
            data_load = np.load(data_pick, allow_pickle=True).item()
            data_load_ = np.load(data_pick_, allow_pickle=True).item()
            src_lang_dir = data_pick.replace(data_pick.split("/")[-1],
                                                        'task_emb_bert.npy')
            tar_lang_dir = data_pick_.replace(data_pick_.split("/")[-1],
                                                        'task_emb_bert.npy')
            # find those common keys
            cam_key = list(data_load.keys())
            cam_key = [cam_k for cam_k in cam_key if cam_k in data_load_.keys()]
            cam_key_select = cam_key[np.random.randint(len(cam_key))]
            src = data_load[cam_key_select]
            target = data_load_[cam_key_select]

        # get the source view 
        if "mp4_dir" not in src.keys():
            print(data_pick, cam_key, src, len(src.keys()))
        src_vr = decord.VideoReader(src['mp4_dir'])
        src_vid = torch.from_numpy(src_vr.get_batch(src["idx"]).asnumpy()).permute(0,3,1,2)
        src_tracks = (torch.from_numpy(src['tracks'])[src["idx"]]-0.5)*2
        src_vis = torch.from_numpy(src['vis'])[src["idx"]]
        # get the target view
        target_vr = decord.VideoReader(target['mp4_dir'])
        fps = (len(target["idx"])//32) if len(target["idx"])>32 else 1
        idx_target = torch.arange(0, len(target["idx"]), fps)
        target_vid = torch.from_numpy(target_vr.get_batch(target["idx"]).asnumpy()).permute(0,3,1,2)
        target_tracks = (torch.from_numpy(target['tracks'])[target["idx"]]-0.5)*2
        target_vis = torch.from_numpy(target['vis'])[target["idx"]]
        target_vid = target_vid[idx_target]
        target_tracks = target_tracks[idx_target]
        target_vis = target_vis[idx_target]
        
        if target_vid.shape[0] < 17:
            target_vid = torch.cat([target_vid, 
                                    target_vid[-1:].repeat(17-target_vid.shape[0],1,1,1) 
                                    ], dim=0)
            target_tracks = torch.cat([target_tracks,
                                    target_tracks[-1:].repeat(17-target_tracks.shape[0],1,1)
                                    ], dim=0)
            target_vis = torch.cat([target_vis,
                                    target_vis[-1:].repeat(17-target_vis.shape[0],1)
                                    ], dim=0)
        try:
            img_t_id = torch.randint(0, target_vid.shape[0]-16, (1,)).item() 
            img_t = target_vid[img_t_id]
            img_t_tracks = target_tracks[img_t_id:img_t_id+16]
            img_t_vis = target_vis[img_t_id:img_t_id+16]
            iters = 0
            while img_t_vis[0].sum()<50:
                img_t_id = torch.randint(0, target_vid.shape[0]-16, (1,)).item() 
                img_t = target_vid[img_t_id]
                img_t_tracks = target_tracks[img_t_id:img_t_id+16]
                img_t_vis = target_vis[img_t_id:img_t_id+16]
                iters+=1
                if iters>5:
                    img_t_vis = torch.ones_like(img_t_vis)
                    break
        except:
            print(target_vid.shape[0])
            raise ValueError
        # get the source condition  
        # pick the key-frame by its velocity
        # velocity = (src_tracks[1:,...]
        #             -src_tracks[:-1,...]).norm(dim=-1).mean(dim=-1)
        # key_frame = torch.sort(velocity, descending=True).indices[:32]
        # key_frame = torch.sort(key_frame).values
        fps = (src_vid.shape[0]//32) if src_vid.shape[0]>32 else 1
        key_frame = torch.arange(0,src_vid.shape[0],fps)[:32]
        imgs_src = src_vid[key_frame]
        tracks_src = src_tracks[key_frame]
        vis_src = src_vis[key_frame]

        if imgs_src.shape[0]==0:
            print(data_pick, cam_key, src, len(src.keys()))
            os.remove(data_pick)
            raise ValueError

        if imgs_src.shape[0]<32:
            imgs_src = torch.cat([imgs_src, 
                                imgs_src[-1:].repeat(32-imgs_src.shape[0],1,1,1) 
                                ], dim=0)
            tracks_src = torch.cat([tracks_src,
                                tracks_src[-1:].repeat(32-tracks_src.shape[0],1,1)
                                ], dim=0)
            vis_src = torch.cat([vis_src,
                                vis_src[-1:].repeat(32-vis_src.shape[0],1)
                                ], dim=0)

        # get the txt embedding
        src_task_emb = torch.from_numpy(np.load(src_lang_dir,
                                                 allow_pickle=True)).float()
        target_task_emb = torch.from_numpy(np.load(tar_lang_dir,
                                                    allow_pickle=True)).float()
        
        # resize the image into the fixed size
        img_t = torch.nn.functional.interpolate(img_t.unsqueeze(0), (140,140)).squeeze(0)
        imgs_src = torch.nn.functional.interpolate(imgs_src, (140,140))

        # filter the tracks whose the first frame is not visible
        msk = img_t_vis[0,:] # 1 N
        img_t_tracks = img_t_tracks[:, msk]
        if msk.sum() < 800:
            idx = torch.randint(0, msk.sum(), (800-msk.sum(),))
            img_t_tracks = torch.cat([img_t_tracks,
                                    img_t_tracks[:, idx]
                                    ], dim=1)

        ret = {'img_tar':img_t, 'tracks_tar':img_t_tracks, 'vis_tar':img_t_vis,
                'txtemb_src':src_task_emb, 'txtemb_tar':target_task_emb,
                'imgs_src':imgs_src, 'tracks_src':tracks_src, 'vis_src':vis_src}
        return ret
