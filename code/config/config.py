# -*- coding: utf-8 -*-

import argparse
import os

def get_parser(prog='tsbir'):
    parser=argparse.ArgumentParser(prog)
    
    parser.add_argument('--gpu_nums',
                        type=int,
                        default=1,
                        help='the number of gpus')
    # dataset
    parser.add_argument('--dataset',
                        required=True,
                        help='the dataset type')
    
    parser.add_argument('--dataset_root_path',
                        required=True,
                        help='the root path for dataset')
    
    parser.add_argument('--loader_num_workers',
                        type=int,
                        default=5,
                        help='the number of loader workers')
    
    parser.add_argument('--class_number',
                        type=int,
                        default=41,
                        help='the class number for a sketch dataset')
    
    parser.add_argument('--logger_comment',
                        type=str,
                        default="tsbir_SFSD_",
                        help='logger name')
    
    parser.add_argument('--logger_path',
                        type=str,
                        default="./runs/",
                        help='logger save path')
    
    # model
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='the batch size')
    
    parser.add_argument('--n_epoch',
                        type=int,
                        default=100,
                        help='the max epoch number')
    
    parser.add_argument('--dropout',
                        type=float,
                        default=0.1,
                        help='the dropout ratio')
    
    parser.add_argument('--resume',
                        type=str,
                        default="./runs/Sep15_18-41-49_cu02tsbir_SFSD_/latest_checkpoint.pth",
                        help='model postion')
    
    #lr_scheduler:
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='optimizer learning_rate')
    
    parser.add_argument('--scheduler',
                        default='StepLR',
                        choices=['StepLR','ReduceLROnPlateau'],
                        help='lr_scheduler type')
    
    parser.add_argument('--learning_rate_decay_frequency',
                        type=int,
                        default=5,
                        help='lr_scheduler learning_rate_decay_frequency')
    
    parser.add_argument('--learning_rate_factor',
                        type=float,
                        default=0.5,
                        help='lr_scheduler learning_rate_factor')
    
    
    
    return parser
