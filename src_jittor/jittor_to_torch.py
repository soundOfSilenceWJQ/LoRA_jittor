import jittor as jt

import argparse
import time
import math
import os, sys
import numpy as np
import itertools

import torch
import random
torch.set_printoptions(threshold=100000)

from gpu import (
    add_gpu_params
)

from data_utils_jittor import FT_Dataset
from model_torch import GPT2Config_torch, GPT2LMModel_torch
from model_jittor import GPT2LMModel as GPT2LMModel_jittor
from model_jittor import GPT2Config as GPT2Config_jittor

from exp_utils_jittor import create_exp_dir

import loralib as lora

def print_args(args):
    print('=' * 100)
    for k, v in args.__dict__.items():
        print(f'        - {k} : {v}')
    print('=' * 100)

parser = argparse.ArgumentParser(description='PyTorch GPT2 ft script')

parser.add_argument('--model_card', default='gpt2.md', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'], 
                    help='model names')

parser.add_argument('--init_checkpoint', default='/root/HW5/LoRA/examples/NLG/trained_models/GPT2_jittor_test/GPT2_MD/e2e/model.100.pkl', help='pretrained checkpoint path')

parser.add_argument('--lora_dim', type=int, default=4, help='lora attn dimension')

parser.add_argument('--lora_alpha', type=int, default=32, help='lora attn alpha')

parser.add_argument('--lora_dropout', default=0.1, type=float, 
                    help='dropout probability for lora layers')

if __name__ == '__main__':
    args = parser.parse_args()
    print_args(args)
    if args.model_card == 'gpt2.sm':
        config = GPT2Config_torch(
            n_embd=768, n_layer=12, n_head=12, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
        config_jittor = GPT2Config_jittor(
            n_embd=768, n_layer=12, n_head=12, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == 'gpt2.md':
        config = GPT2Config_torch(
            n_embd=1024, n_layer=24, n_head=16, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
        config_jittor = GPT2Config_jittor(
            n_embd=1024, n_layer=24, n_head=16, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == 'gpt2.lg':
        config = GPT2Config_torch(
            n_embd=1280, n_layer=36, n_head=20, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
        config_jittor = GPT2Config_jittor(
            n_embd=1280, n_layer=36, n_head=20, 
            lora_attn_dim=args.lora_dim, 
            lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
    
    
    # lm_net_jittor = GPT2LMModel_jittor(config_jittor)
    # # 初始化一个线性层
    # # lm_net_jittor = jt.nn.Linear(1280, 1280)
    # # 将lm_net_jittor保存
    # # lm_net_jittor.save("/root/HW5/LoRA/examples/NLG/trained_models/lm_net_jittor.pt")
    # lm_net_jittor.load( "/root/HW5/LoRA/examples/NLG/trained_models/GPT2_jittor_test/GPT2_MD/e2e/model.100.pkl")
    # # print(lm_net_jittor)
    # jittor_weights = lm_net_jittor.state_dict()
    lm_net = GPT2LMModel_jittor(config_jittor)
    # torch.load(lm_net, "/root/HW5/LoRA/examples/NLG/trained_models/GPT2_jittor_test/GPT2_MD/e2e/model.100.pkl")
    lm_net.load("/root/HW5/LoRA/examples/NLG/trained_models/GPT2_jittor/GPT2_MD/e2e/model.70000.pt")
    # lm_net.load(cp)
    # lm_net.cuda()
    # lm_net.load_state_dict(jittor_weights)
    # print(lm_net)
