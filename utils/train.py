#!/usr/bin/env python
# coding: utf-8

import os
import time
# import math
import torch
# import pickle
import argparse

import numpy as np
import os.path as osp
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

from layers import *
from models import *
from preprocessing import *
from dataloader import load_data
import warnings
from model_loader import parse_method
from model_data_parser import load_hnn_parser
from inference import train_epoch, infer_epoch
from metrics import Eval_Metrics, Eval_Metrics_Average

warnings.filterwarnings("ignore")


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            return best_result[:, 1], best_result[:, 3]

    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
#             print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])


@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out, _, _ = model(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        data.y[split_idx['test']], out[split_idx['test']])

    train_loss = F.nll_loss(
        out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = F.nll_loss(
        out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss = F.nll_loss(
        out[split_idx['test']], data.y[split_idx['test']])
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct))/len(correct))
    return sum(acc_list)/len(acc_list)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Main part of the training ---
# # Part 0: Parse arguments

if __name__ == '__main__':
    parser = load_hnn_parser()

    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--wandb', action='store_true', default=False, help='Use wandb for logging')
    parser.add_argument('--save_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--teacher_epochs', type=int, default=500, help='training epochs')
    parser.add_argument('--num_experts', type=int, default=1, help='training iterations')
    parser.add_argument('--param_save_interval', type=int, default=1)
    parser.add_argument('--traj_save_interval', type=int, default=10)
    
    args = parser.parse_args()

    set_seed(args.seed)
    
    # # Part 2: Load model

    data, split_idx_lst = load_data(args)
    
    model = parse_method(args, data)
    # put things to device
    if args.cuda in [0, 1]:
        device = torch.device('cuda:'+str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    model, data = model.to(device), data.to(device)
    if args.method == 'UniGCNII':
        args.UniGNN_degV = args.UniGNN_degV.to(device)
        args.UniGNN_degE = args.UniGNN_degE.to(device)
    
    num_params = count_parameters(model)

    results = Eval_Metrics_Average()
    
    # # Part 3: Main. Training + Evaluation
    
    criterion = nn.NLLLoss()
    eval_func = eval_acc

    logger = Logger(args.runs)
    
    model.train()
    
    ### Training loop ###
    for run in tqdm(range(args.runs)):
        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        model.reset_parameters()
        if args.method == 'UniGCNII':
            optimizer = torch.optim.Adam([
                dict(params=model.reg_params, weight_decay=0.01),
                dict(params=model.non_reg_params, weight_decay=5e-4)
            ], lr=0.01)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        best_val = float('-inf')
        for epoch in range(args.epochs):
            #         Training part
            # model.train()
            # optimizer.zero_grad()
            # out, _, _ = model(data)
            # out = F.log_softmax(out, dim=1)
            # loss = criterion(out[train_idx], data.y[train_idx])
            # loss.backward()
            # optimizer.step()
            train_epoch(model, data, train_idx, optimizer, epoch)
            result = evaluate(model, data, split_idx, eval_func)
            logger.add_result(run, result[:3])

            val_res = infer_epoch(model, data, split_idx['valid'].to(device))
            print(f"val_acc: {val_res.acc}, val_macro_f1: {val_res.macro_f1}, val_micro_f1: {val_res.micro_f1}")
    
            # if epoch % args.display_step == 0 and args.display_step > 0:
            #     print(f'Epoch: {epoch:02d}, '
            #           f'Train Loss: {loss:.4f}, '
            #           f'Valid Loss: {result[4]:.4f}, '
            #           f'Test  Loss: {result[5]:.4f}, '
            #           f'Train Acc: {100 * result[0]:.2f}%, '
            #           f'Valid Acc: {100 * result[1]:.2f}%, '
            #           f'Test  Acc: {100 * result[2]:.2f}%')

    best_val, best_test = logger.print_statistics()
