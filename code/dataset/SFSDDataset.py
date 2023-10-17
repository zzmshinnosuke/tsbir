#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on 2023-09-11 10:50:39
# @Author: zzm

from PIL import Image
import numpy as np
import glob
import os
import json

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from code.clip import _transform, tokenize

MAX_LENGTH = 77
input_resolution = 224

class SFSDDataset(Dataset):
    def __init__(self, config, split = "train"):
        self.config = config
        self.split = split
        self.root_path = config.dataset_root_path
        self.sketch_path = os.path.join(self.root_path, "sketches")
        self.images_path = os.path.join(self.root_path, "images")
        self.sketchImg_path = os.path.join(self.root_path, "sketchImgs")
        self._transform = _transform(input_resolution, is_train=False)
        self.files = list()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.load_files_path()
        self.load_categories()

    def load_categories(self):
        file = os.path.join(self.root_path, "categories_info.json")
        with open(file, 'r') as fp:
            self.categories_info = json.load(fp)

    def load_files_path(self):
        assert self.split in ['train', 'test', 'traintest'], 'unknown split {}'.format(self.split)

        if self.split == 'traintest':
            self.files = glob.glob(os.path.join(self.root_path, 'sketch', '*.json'))
        else: 
            filename_txt = 'train_names.txt' if self.split == 'train' else 'test_names.txt'
            filename_path = os.path.join(self.root_path, filename_txt)
            assert os.path.exists(filename_path), 'not find {}'.format(filename_path)
            with open(filename_path, 'r') as f:
                self.files = [line.strip() for line in f.readlines()]
        assert len(self.files)>0, 'no sketch json file find in {}'.format(self.root_path)

        captionpath = os.path.join(self.root_path, self.split+'.json')
        with open(captionpath, "r") as f:
            try:
                self.all_captions_cats = json.load(f)
            except json.decoder.JSONDecodeError:
                print("don't have "+ captionpath)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sketch_id = self.files[index]

        image_path = os.path.join(self.images_path, sketch_id + ".jpg")
        sketch_path = os.path.join(self.sketchImg_path, sketch_id + ".png")
        image = Image.open(image_path)
        sketch = Image.open(sketch_path)
        image_tran = self._transform(image)
        sketch_tran = self._transform(sketch)
        
        cate = torch.tensor(np.array(list(self.all_captions_cats[sketch_id]['cats'])))
        
        caption = self.all_captions_cats[sketch_id]['captions'][0]
        tokenized = self.tokenizer.encode("<|endoftext|> " + caption + " <|endoftext|>")[:MAX_LENGTH]
        masks = torch.zeros(MAX_LENGTH)
        masks[torch.arange(len(tokenized))] = 1
        tokens = torch.zeros(MAX_LENGTH).long()
        tokens[torch.arange(len(tokenized))] = torch.LongTensor(tokenized)

        txt = tokenize([str(caption)])[0]

        return image_tran, sketch_tran, txt, cate , tokens, masks

        