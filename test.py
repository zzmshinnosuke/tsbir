#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2023-09-11 00:20:09
# @Author: zzm
import json
from tqdm import tqdm
import numpy as np

import torch
from sklearn.neighbors import NearestNeighbors

from code.dataset import get_loader
from code.clip import convert_weights, CLIP, tokenize
from code.config import get_parser

def test(test_dataloader, clipmodel):
    with torch.no_grad():
        recall = 0
        img_feats = []
        fused_feats = [] 
        for batch in tqdm(test_dataloader):
            image, sketch, txt, _, _, _, = batch
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

'''
python test.py --dataset SFSDDataset --dataset_root_path ~/datasets/SFSD
'''
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    test_dataloader = get_loader(args, 'test')

    Top_K = 1

    model_config_file = './code/training/model_configs/ViT-B-16.json'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)
    model = CLIP(**model_info)

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    clipmodel = model.to(device)
    convert_weights(clipmodel)
    test(test_dataloader, clipmodel)