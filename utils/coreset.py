import os
import sys
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from copy import deepcopy
import numpy as np
import logging
import random
import argparse

from model.model_loader import parse_method
from utils import init_logger, init_parameters, init_wandb, resolve_device, to_device, filter_hyperedge
from dataloader import load_data
from coreset_utils import KCenter, Herding, Random
from tqdm import tqdm
import torch
import datetime
import torch.nn.functional as F
from inference import train_epoch, infer_epoch
# from model_loader import parse_method
from metrics import Eval_Metrics_Average
from model_data_parser import load_hnn_parser

import warnings
warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def select_coreset(data, split_idx_lst, log_dir, device, args):
    if args.method == 'UniGCNII':
        args.UniGNN_degV = args.UniGNN_degV.to(device)
        args.UniGNN_degE = args.UniGNN_degE.to(device)
    
    for run, split_idx in enumerate(split_idx_lst):
        train_idx, val_idx, test_idx = split_idx['train'].to(device), \
            split_idx['valid'].to(device), split_idx['test'].to(device)

        net = parse_method(args, data).to(device)

        # 根据所选方法，使用不同的优化器来训练模型
        if args.method == 'UniGCNII':
            optimizer = torch.optim.Adam([
                dict(params=net.reg_params, weight_decay=0.01),
                dict(params=net.non_reg_params, weight_decay=5e-4)
            ], lr=0.01)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
        best_eval, best_epoch, best_state = 0, 0, None
        
        for e in range(args.epochs + 1):
            # 在指定的 epochs 中进行训练，计算并记录训练集的损失
            loss = train_epoch(net, data, train_idx, optimizer, e, return_emb=False)
            logging.info(f'Epochs={e}: Full graph train set results: loss = {loss:.4f}')
            
            # 每个 epoch 后，计算验证集的准确率，并保存最好的模型参数
            res = infer_epoch(net, data, val_idx, return_emb=False)
            if res.acc > best_eval:
                best_eval = res.acc
                best_epoch = e
                best_state = deepcopy(net.state_dict())
            logging.info(f'Epochs={e} accuracy = {res.acc:.4f}')
        
        # 在训练结束后，使用最佳模型在测试集上进行评估，并记录结果
        logging.info(f'best epoch={best_epoch}, best eval={best_eval:.4f}')
        net.load_state_dict(best_state)
        res = infer_epoch(net, data, test_idx, test=True, return_emb=False)
        logging.info(f"Test set results: Full graph train set acc = {res.acc:.4f}")
        logging.info(f"{res}")

        # 获得训练集的节点嵌入（emb）和超边嵌入（emb_edge）
        _, emb, emb_edge = infer_epoch(net, data, train_idx, return_emb=True)

        if args.core_method == 'kcenter':
            agent = KCenter(data, train_idx, args, device=device)
        if args.core_method == 'herding':
            agent = Herding(data, train_idx, args, device=device)
        if args.core_method == 'random':
            agent = Random(data, train_idx, args, device=device)
        idx_selected = agent.select(emb)
        if args.save:
            logging.info('Saving...')
            np.save(f'{log_dir}/idx_{args.dname}_{args.reduction_rate}_{args.core_method}_{args.seed}_{run}.npy', idx_selected.cpu().numpy())
        logging.info(args)
        logging.info(log_dir)

if __name__ == "__main__":
    parser = load_hnn_parser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save', action='store_true', default=True)
    parser.add_argument('--lr_coreset', type=float, default=0.005)
    parser.add_argument('--wd_coreset', type=float, default=0)
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--coreset_epochs', type=int, default=100)
    parser.add_argument('--save_log', type=str, default='logs_AllDeepSets')
    parser.add_argument('--core_method', type=str, default='herding', choices=['kcenter', 'herding', 'random'])
    parser.add_argument('--reduction_rate', type=float, default=0.03)
    args = parser.parse_args()

    device = resolve_device(args.device)
    log_dir = './' + args.save_log + '/Coreset/{}-reduce_{}-{}'.format(args.dname, str(args.reduction_rate), args.core_method)

    init_logger(args, log_dir)

    # random seed setting
    set_seed(args.seed)
    data, split_idx_lst = load_data(args)

    data = data.to(device)

    select_coreset(data, split_idx_lst, log_dir, device, args)
