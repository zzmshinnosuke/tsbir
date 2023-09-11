#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2023-09-11 10:50:39
# @Author: zzm
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from transformers.modeling_utils import *
from transformers import GPT2Tokenizer

from code.clip import _transform, tokenize

MAX_LENGTH = 77
input_resolution = 224

class SFSDDataset(Dataset):
    # tfms = _transform(clipmodel.visual.input_resolution, is_train=True)
    tfms = _transform(input_resolution, is_train=False)
    
    def __init__(self, images_path, sketch_path, cc, random_transform=False):
        self.images_path = images_path
        self.sketch_path = sketch_path
        self.cc = cc
        self.random_transform = random_transform
        
    def __len__(self):
        return len(self.cc)

    def __getitem__(self, index):
        image_id = self.cc[index][0]
        text = self.cc[index][1]
        category = self.cc[index][2]
        
        # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # tokenized = tokenizer.encode("<|endoftext|> " + text + " <|endoftext|>")[:MAX_LENGTH]
        # masks = torch.zeros(MAX_LENGTH)
        # masks[torch.arange(len(tokenized))] = 1
        # tokens = torch.zeros(MAX_LENGTH).long()
        # tokens[torch.arange(len(tokenized))] = torch.LongTensor(tokenized)

        image_path = os.path.join(self.images_path, image_id + ".jpg")
        sketch_path = os.path.join(self.sketch_path, image_id + ".jpeg")
        image = Image.open(image_path)
        sketch = Image.open(sketch_path)
        image_tran = self.tfms(image)
        sketch_tran = self.tfms(sketch)
        # cate = torch.tensor(category)
        cate = torch.tensor(np.array(category))
        
        return image_id, image_tran, sketch_tran, text, cate #, tokens, masks
    