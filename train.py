#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2023-09-11 00:19:59
# @Author: zzm

import os
import time
import pickle
import json
from collections import namedtuple

from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.checkpoint as checkpoint
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# from transformers.modeling_utils import *
# from transformers.modeling_gpt2 import *
from transformers.models.gpt2.modeling_gpt2 import GPT2Config
from transformers import GPT2Tokenizer
from transformers import AdamW

from .code.clip.clip import _transform, load, tokenize
from .code.clip.model import convert_weights, CLIP
from .code.AsymmetricLoss import AsymmetricLossOptimized, AsymmetricLoss
from .code.dataset.CaptionDataset import CaptionDataset
from .code.gpt.gpt import GPT2LMHeadModel

MAX_LENGTH = 77
EFFNET_OUT = 512

model_config_file = './model_configs/ViT-B-16.json'
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
# model.eval()
model.train()
clipmodel = model.to(device)
convert_weights(clipmodel)

TRAIN_CC = "data/train_cc.json"


def load_cc(path):
    with open(path, "rb") as f:
        ccs = json.load(f)
    cc2 = []
    for i, cc in enumerate(ccs):
        cc2.append((cc[0], cc[1], cc[2]))
    return cc2

train_cc = load_cc(TRAIN_CC)
test_cc = load_cc(TEST_CC)


BS = 4
image_path = '../../Data/SFSD/images/'
sketch_path = '../../Data/SFSD/sketchImg'
# train_dataloader = ld_train.load_data(image_path, sketch_path, batch_size)
# test_dataloader = ld_test.load_data(image_path, sketch_path, batch_size)

train_dataset = CaptionDataset(image_path, sketch_path, train_cc)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BS, shuffle=True, drop_last=True)


epoch = 50
loss_img = nn.CrossEntropyLoss().to(device)
loss_txt_sketch = nn.CrossEntropyLoss().to(device)
optimizer = AdamW(clipmodel.parameters(), lr=1e-6)

# 定义模型
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).cuda()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim).cuda()
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        # self.sigmoid = nn.Softmax()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # out = self.sigmoid(out)
        out = self.relu(out)
        return out

input_dim = 512
hidden_dim = 256
output_dim = 40

model_file_text = './model_pt/15_lc_text.pt'
model_file_sketch = './model_pt/26_lc_sketch.pt'
model_file_image = './model_pt/27_lc_image.pt'

modelText = Net(input_dim, hidden_dim, output_dim)
modelText.load_state_dict(torch.load(model_file_text, map_location='cpu'), strict=False)
modelText.eval()
# model.train()
modelTextLc = modelText.to(device)
convert_weights(modelTextLc)

modelSketch = Net(input_dim, hidden_dim, output_dim)
modelSketch.load_state_dict(torch.load(model_file_sketch, map_location='cpu'), strict=False)
modelSketch.eval()
modelSketchLc = modelSketch.to(device)
convert_weights(modelSketchLc)

modelImage = Net(input_dim, hidden_dim, output_dim)
modelImage.load_state_dict(torch.load(model_file_image, map_location='cpu'), strict=False)
modelImage.eval()
modelImageLc = modelImage.to(device)
convert_weights(modelImageLc)

model_file = './model_pt/29_ld.torch'
config = GPT2Config(
    vocab_sizea = 49408,
    n_layer=6,
    n_head=8,
    n_ctx=77,
)
model = GPT2LMHeadModel(config).cuda()
# checkpoints = torch.load(model_file, map_location='cpu')
model.load_state_dict(torch.load(model_file, map_location='cpu'), strict=False)
model.eval()
# model.train()
ldmodel = model.to(device)
convert_weights(ldmodel)

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

recall_last = 0
for i in range(epoch):
    step = 0
    for batch in tqdm(train_dataloader):
        step += 1
        image_id, image, sketch, txt, cate, tokens, masks= batch
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
            ground_truth = torch.arange(BS).long().to(device)
        else:
            ground_truth = torch.arange(BS, dtype=torch.long, device=device)
        Le_loss = (loss_img(logits_per_image, ground_truth) + loss_txt_sketch(logits_per_fuse, ground_truth)) / 2

        #Lc
        ASL_Loss = AsymmetricLossOptimized()
        logit_txt = modelTextLc(text_feature)
        logit_img = modelImageLc(image_feature) #应为三个model
        logit_sketch = modelSketchLc(sketch_feature)
        Lc_loss_txt = ASL_Loss(logit_txt, cate)
        Lc_loss_img = ASL_Loss(logit_img, cate)
        Lc_loss_sketch = ASL_Loss(logit_sketch, cate)
        Lc_loss = (Lc_loss_txt + Lc_loss_img + Lc_loss_sketch) / (3 * BS)         
        
        #Ld
        with torch.no_grad():
            Ld_loss, outputs, _ = ldmodel(tokens, fused_feature, labels=tokens, attention_mask=masks)


        total_loss = (10 * Lc_loss + Ld_loss + 100 * Le_loss) / 111
        # total_loss = (Ld_loss + 100 * Le_loss) / 101
        # total_loss = Le_loss

        
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            # convert_models_to_fp32(clipmodel)
            optimizer.step()
            convert_weights(clipmodel)
        optimizer.zero_grad()
        
        if step % 100 == 0:
            print('[%d / %d] loss: %.10f' %(i + 1, step, total_loss))
        
    torch.save(clipmodel.state_dict(), "total" + str(i) + ".pt")

clipmodel.eval()
torch.save(clipmodel.state_dict(), "lr1e-10_le.pt")