
import os, sys
import numpy as np
import glob
import tqdm

clean_samples = list(np.load("./cleaned_data.npy")) if os.path.exists("./cleaned_data.npy") else []
data_file_path = glob.glob('/nas3/xyx/ILAnyDATA/*/*.npy')
data_file_path = [dirs for dirs in data_file_path if ('task_emb_bert' not in dirs)&(dirs not in clean_samples)]
del_cam_num = 0
cleaned_data = []
for file_path in tqdm.tqdm(data_file_path):
    data = np.load(file_path, allow_pickle=True)
    cam_key = data.item().keys()
    data_dict = data.item().copy()
    if len(cam_key) < 2:
        os.remove(file_path)
        continue
    for key in cam_key:
        if (len(data_dict[key].keys()) == 0):
            data_dict.pop(key)
            del_cam_num += 1
        else:
            if len(data_dict[key]["idx"]) < 10:
                data_dict.pop(key)
                del_cam_num += 1
    np.save(file_path, data_dict)
    cleaned_data.append(file_path)
print('Delete camera number:', del_cam_num)
np.save('./cleaned_data.npy', cleaned_data)
