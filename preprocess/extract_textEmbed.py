
import sys
import torch
import os
import math
from tqdm import tqdm
from glob import glob
import h5py
from easydict import EasyDict
import numpy as np
from multiprocessing import Process
from transformers import AutoTokenizer, AutoModel
import json

def get_task_embs(cfg, descriptions):
    if cfg.task_embedding_format == "one-hot":
        # offset defaults to 1, if we have pretrained another model, this offset
        # starts from the pretrained number of tasks + 1
        offset = cfg.task_embedding_one_hot_offset
        descriptions = [f"Task {i+offset}" for i in range(len(descriptions))]

    if cfg.task_embedding_format == "bert" or cfg.task_embedding_format == "one-hot":
        tz = AutoTokenizer.from_pretrained(
            "/nas1/checkpoints/huggingface/models--bert-base-uncased"
        )
        model = AutoModel.from_pretrained(
            "/nas1/checkpoints/huggingface/models--bert-base-uncased"
        )
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        masks = tokens["attention_mask"]
        input_ids = tokens["input_ids"]
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])[
            "pooler_output"
        ].detach()
    elif cfg.task_embedding_format == "gpt2":
        tz = AutoTokenizer.from_pretrained("gpt2")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("gpt2")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["last_hidden_state"].detach()[:, -1]
    elif cfg.task_embedding_format == "clip":
        tz = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model.get_text_features(**tokens).detach()
    elif cfg.task_embedding_format == "roberta":
        tz = AutoTokenizer.from_pretrained("roberta-base")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("roberta-base")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["pooler_output"].detach()
    cfg.policy.language_encoder.network_kwargs.input_size = task_embs.shape[-1]
    return task_embs


def get_task_name_from_file_name(file_name):
    name = file_name.replace('_demo', '')
    if name[0].isupper():  # LIBERO-100
        if "SCENE10" in name:
            language = " ".join(name[name.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(name[name.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(name.split("_"))
    return language


def add_task_emb(h5_file_list, task_name_to_emb, skip_exist=False):
    for h5_file in tqdm(h5_file_list):
        print(h5_file)
        demo_h5 = h5py.File(h5_file, 'a')
        root_grp = demo_h5["root"]

        if ("task_emb_bert" in root_grp) and skip_exist:
            continue
        task_name = get_task_name_from_file_name(h5_file.split('/')[-2])
        emb = task_name_to_emb[task_name]

        if "task_emb_bert" in root_grp:
            root_grp.__delitem__("task_emb_bert")
        root_grp.create_dataset("task_emb_bert", data=emb)
        demo_h5.close()


def main():
    
    ROOT_DIR = "/nas3/xyx/ILAnyDATA/"
    JSON_DIR = f"/nas3/xyx/RH20T/task_description.json"

    with open(JSON_DIR, 'r') as f:
        task_desc = json.load(f)

    tasks_list = os.listdir(ROOT_DIR)

    # set the task embeddings  
    cfg = EasyDict({
        "task_embedding_format": "bert",
        "task_embedding_one_hot_offset": 1,
        "data": {"max_word_len": 25},
        "policy": {"language_encoder": {"network_kwargs": {"input_size": 768}}}
    })  # hardcode the config to get task embeddings according to original Libero code

    for task in tqdm(tasks_list):
        text_input = task_desc[task]["task_description_english"]
        task_embs = get_task_embs(cfg, text_input).cpu().numpy()
        save_path = os.path.join(ROOT_DIR, task, "task_emb_bert.npy")
        np.save(save_path, task_embs)

if __name__ == '__main__':
    main()
