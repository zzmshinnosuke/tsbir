#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2023-09-11 00:20:09
# @Author: zzm
import json
from tqdm import tqdm
import numpy as np

import torch
from sklearn.neighbors import NearestNeighbors

from code.dataset import SFSDDataset
from code.clip import convert_weights, CLIP, tokenize

Top_K = 1

TEST_CC = "data/test_cc.json"
image_path = './SFSD/images/'
sketch_path = './SFSD/sketchImg'

def load_cc(path):
    with open(path, "rb") as f:
        ccs = json.load(f)
    cc2 = []
    for i, cc in enumerate(ccs):
        cc2.append((cc[0], cc[1], cc[2]))
    return cc2

test_cc = load_cc(TEST_CC)

test_dataset = SFSDDataset(image_path, sketch_path, test_cc)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=5)

model_config_file = './code/training/model_configs/ViT-B-16.json'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open(model_config_file, 'r') as f:
    model_info = json.load(f)
model = CLIP(**model_info)
model_file = 'total14.pt'
model.load_state_dict(torch.load(model_file, map_location='cpu'), strict=False)
model.eval()
clipmodel = model.to(device)
convert_weights(clipmodel)

with torch.no_grad():
    recall = 0
    total = 0
    img_feats = []
    fused_feats = [] 
    for batch in tqdm(test_dataloader):
        _, image, sketch, txt, _ = batch
        image = image.cuda()
        sketch = sketch.cuda()
        txt = tokenize([str(txt)])[0].unsqueeze(0).to(device)
        image_feature, fused_feature = clipmodel(image, txt, sketch)
        img_feats.append(image_feature.cpu().detach().numpy()[0])
        fused_feats.append(fused_feature.cpu().detach().numpy())

    nbrs = NearestNeighbors(n_neighbors=Top_K, algorithm='brute', metric='cosine').fit(img_feats)
    for index,ff in enumerate(fused_feats):
        distances, indices = nbrs.kneighbors(ff)
        for ind in indices:
            if index in ind:
                recall += 1
    print(round(recall / len(fused_feats), 4))