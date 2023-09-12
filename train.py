#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2023-09-11 00:19:59
# @Author: zzm

import os
import time
import pickle
import json
from collections import namedtuple

from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
from torch.optim import AdamW

from transformers.models.gpt2.modeling_gpt2 import GPT2Config
# from transformers import AdamW

from code.clip import tokenize, convert_weights, CLIP, ClassModel
from code.AsymmetricLoss import AsymmetricLossOptimized
from code.gpt import GPT2LMHeadModel
from code.dataset import get_loader
from code.config import get_parser

MAX_LENGTH = 77
EFFNET_OUT = 512

input_dim = 512
hidden_dim = 256
output_dim = 40

def train(args, train_dataloader, clipmodel, gptmodel, classmodel):
    loss_img = nn.CrossEntropyLoss().to(device)
    loss_txt_sketch = nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(clipmodel.parameters(), lr=1e-6)
    ASL_Loss = AsymmetricLossOptimized()
    for i in range(args.n_epoch):
        step = 0
        for batch in tqdm(train_dataloader):
            step += 1
            image, sketch, txt, cate, tokens, masks = batch
            image, sketch, cate, tokens, masks = image.cuda(), sketch.cuda(), cate.cuda(), tokens.cuda(), masks.cuda()
            txt = tokenize([str(txt)])[0].unsqueeze(0).to(device)
            image_feature, fused_feature = clipmodel(image, txt, sketch)
            sketch_feature = clipmodel.encode_sketch(sketch)
            text_feature = clipmodel.encode_text(txt)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            sketch_feature = sketch_feature / sketch_feature.norm(dim=-1, keepdim=True)

            #Le
            logit_scale = clipmodel.logit_scale.exp()
            logits_per_image = logit_scale * image_feature @ fused_feature.t()
            logits_per_fuse = logits_per_image.t()
            if device == "cpu":
                ground_truth = torch.arange(batch[0].shape[0]).long().to(device)
            else:
                ground_truth = torch.arange(batch[0].shape[0], dtype=torch.long, device=device)
            Le_loss = (loss_img(logits_per_image, ground_truth) + loss_txt_sketch(logits_per_fuse, ground_truth)) / 2

            #Lc
            logit_txt = classmodel(text_feature.float())
            logit_img = classmodel(image_feature.float()) #应为三个model
            logit_sketch = classmodel(sketch_feature.float())
            Lc_loss_txt = ASL_Loss(logit_txt, cate)
            Lc_loss_img = ASL_Loss(logit_img, cate)
            Lc_loss_sketch = ASL_Loss(logit_sketch, cate)
            Lc_loss = (Lc_loss_txt + Lc_loss_img + Lc_loss_sketch) / (3 * args.batch_size)         
            
            #Ld
            with torch.no_grad():
                fused_feature = fused_feature.float()
                # print(tokens.dtype, fused_feature.dtype, masks.dtype)
                Ld_loss, outputs, _ = gptmodel(tokens, fused_feature, labels=tokens, attention_mask=masks)

            total_loss = (10 * Lc_loss + Ld_loss + 100 * Le_loss) / 111
            
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else:
                optimizer.step()
                convert_weights(clipmodel)
            optimizer.zero_grad()
            
            if step % 100 == 0:
                print('[%d / %d] loss: %.10f' %(i + 1, step, total_loss))
            
        torch.save(clipmodel.state_dict(), "total" + str(i) + ".pt")

    clipmodel.eval()
    torch.save(clipmodel.state_dict(), "lr1e-10_le.pt")

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_dataloader = get_loader(args, 'train')

    # load clip model
    model_config_file = './code/training/model_configs/ViT-B-16.json'
    model_file = './model_pt/tsbir_model_final.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(model_config_file, 'r') as f:
        model_info = json.load(f)
    model = CLIP(**model_info)
    checkpoints = torch.load(model_file, map_location='cpu')
    sd = checkpoints["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.train()
    clipmodel = model.to(device)

    model_file = './model_pt/29_ld.torch'
    config = GPT2Config(
        vocab_sizea = 49408,
        n_layer=6,
        n_head=8,
        n_ctx=77,
    )
    model = GPT2LMHeadModel(config).cuda()
    model.load_state_dict(torch.load(model_file, map_location='cpu'), strict=False)
    model.eval()
    # model.train()
    ldmodel = model.to(device)

    for name, param in clipmodel.named_parameters():
        for i in range(11):
            s1 = "visual.transformer.resblocks." + str(i) + "."
            if s1 in name:
                param.requires_grad = False
            s2 = "visual2.transformer.resblocks." + str(i) + "."
            if s2 in name:
                param.requires_grad = False
            s3 = "transformer.resblocks." + str(i) + "."
            if s3 in name:
                param.requires_grad = False

    classmodel = ClassModel(input_dim, hidden_dim, output_dim)
    classmodel.eval()
    classmodel = classmodel.to(device)

    train(args, train_dataloader, clipmodel, ldmodel, classmodel)

'''
python train.py --dataset SFSDDataset --dataset_root_path ~/datasets/SFSD --batch_size 8
'''