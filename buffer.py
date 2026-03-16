import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import numpy as np
import random
import argparse
import torch
import torch.nn.functional as F
import logging
from tensorboardX import SummaryWriter
import deeprobust.graph.utils as utils
from utils.dataloader import load_data
from model.model_loader import parse_method
from utils.utils import sort_training_nodes_bipartite, sort_nodes_by_hyperedge_difficulty, training_scheduler, sort_nodes_by_H_matrix
from torch_geometric.data import Data
from meta_gradient import add_norm_to_data
from model_data_parser import load_hnn_parser

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def main(args):
    # 随机种子设置
    random.seed(args.seed_teacher)
    np.random.seed(args.seed_teacher)
    torch.manual_seed(args.seed_teacher)
    torch.cuda.manual_seed(args.seed_teacher)
    device = torch.device(args.device)

    data, split_idx_lst = load_data(args)
    split_idx = random.choice(split_idx_lst)
    train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    features, edge_index, labels = data.x, data.edge_index, data.y

    data_all = Data(
            x = features.clone().detach().to(torch.float),
            edge_index = edge_index.clone().detach().to(torch.long),
            y = labels.clone().detach().to(torch.long)
        ).to(device)
    data_all = add_norm_to_data(data_all)
    
    if args.method == 'UniGCNII':
        args.UniGNN_degV = args.UniGNN_degV.to(device)
        args.UniGNN_degE = args.UniGNN_degE.to(device)

    trajectories = [] # 初始化轨迹列表

    # ------------------------- 课程学习 / 难易度排序 -------------------------
    if args.difficulty_type == 'random':
        # 消融实验：无课程学习，训练节点随机顺序
        sorted_trainset = train_idx.clone().cpu().numpy()
        rng = np.random.default_rng(args.seed_teacher)
        rng.shuffle(sorted_trainset)
        logging.info("[Ablation] Using RANDOM curriculum: training nodes have been shuffled.")

    else:
        # 根据模型类型选择不同的排序方法
        if args.method in ['HGNN', 'HyperGCN']:
            # 对于HGNN、HCHA和HyperGCN模型，使用保存的原始V2E边索引进行排序
            if hasattr(data, 'V2E_edge_index'):
                edge_ref = data.V2E_edge_index
            else:
                logging.info(f"{args.method}模型没有找到V2E_edge_index，使用标准 edge_index 进行排序")
                edge_ref = data.edge_index

            if args.difficulty_type == 'node':
                sorted_trainset = sort_training_nodes_bipartite(edge_ref, labels, data.num_nodes, train_idx, device)
            elif args.difficulty_type == 'hyperedge':
                sorted_trainset = sort_nodes_by_hyperedge_difficulty(edge_ref, labels, data.num_nodes, train_idx, device)

        else:
            # 对于其他模型，使用原有的排序方法
            if args.difficulty_type == 'node':
                sorted_trainset = sort_training_nodes_bipartite(data.edge_index, labels, data.num_nodes, train_idx, device)
            elif args.difficulty_type == 'hyperedge':
                sorted_trainset = sort_nodes_by_hyperedge_difficulty(data.edge_index, labels, data.num_nodes, train_idx, device)

    # 开始训练多个专家模型
    for it in range(0, args.num_experts):
        logging.info(
            '======================== {} -th number of experts for {}-model_type=============================='.format(
                it, args.method))

        model = parse_method(args, data)

        model.apply(weights_init)
        model = model.to(device)

        model_parameters = list(model.parameters()) # 获取参数列表

        if args.method == 'UniGCNII':
            optimizer = torch.optim.Adam([
                dict(params=model.reg_params, weight_decay=0.01),
                dict(params=model.non_reg_params, weight_decay=5e-4)
            ], lr=0.01)
        elif args.method in ['HGNN', 'HCHA', 'HyperGCN']:
            # 为HGNN、HCHA和HyperGCN模型使用更高的学习率
            optimizer = torch.optim.Adam(model.parameters(), 0.01, weight_decay=5e-4)
        else:
            optimizer = torch.optim.Adam(model.parameters(), args.lr_teacher, weight_decay=args.wd)


        timestamps = [] # 初始化参数快照列表

        timestamps.append([p.detach().cpu() for p in model.parameters()]) # 保存初始参数

        best_val_acc = best_test_acc = best_it = 0 # 初始化最优精度记录

        # 设定学习率衰减的时间点
        lr_schedule = [args.teacher_epochs // 2 + 1]

        lr = args.lr_teacher
        lam = float(args.lam)
        T = float(args.T)
        args.lam = lam
        args.T = T 
        scheduler = args.scheduler
        # 正式训练
        for e in range(args.teacher_epochs + 1):
            model.train()
            optimizer.zero_grad()
            output = model.forward(data_all)            
            if isinstance(output, tuple):
                output = output[0] 
            output = F.log_softmax(output, dim=1)
            labels = labels.to(output.device)
            # 选择训练数据（如果是 GEOM 采用排序训练）
            size = training_scheduler(args.lam, e, T, scheduler)

            training_subset = sorted_trainset[:int(size * sorted_trainset.shape[0])]

            loss_buffer = F.nll_loss(output[training_subset], labels[training_subset])
            acc_buffer = utils.accuracy(output[train_idx], labels[train_idx])
            writer.add_scalar('buffer_train_loss_curve', loss_buffer.item(), e)
            writer.add_scalar('buffer_train_acc_curve', acc_buffer.item(), e)
            logging.info("Epochs: {} : Full graph train set results: loss= {:.4f}, accuracy= {:.4f} ".format(e,
                                                                                                             loss_buffer.item(),
                                                                                                             acc_buffer.item()))
            loss_buffer.backward() # 反向传播
            optimizer.step() # 更新参数

            # 若到达学习率衰减点
            if e in lr_schedule and args.decay:
                lr = lr*args.decay_factor
                logging.info('NOTE! Decaying lr to :{}'.format(lr))
                if args.method in ['HGNN', 'HCHA', 'HyperGCN']:
                    # 为HGNN、HCHA和HyperGCN模型使用较高的学习率
                    optimizer = torch.optim.Adam(model_parameters, lr=max(lr, 0.001),
                                                weight_decay=args.wd_teacher)
                elif args.optim == 'SGD':
                    optimizer = torch.optim.SGD(model_parameters, lr=lr, momentum=args.mom_teacher,weight_decay=args.wd_teacher)
                elif args.optim == 'Adam':
                    optimizer = torch.optim.Adam(model_parameters, lr=lr,
                                                       weight_decay=args.wd_teacher)

                optimizer.zero_grad()

            # 每20轮评估一次验证集和测试集
            if e % 20 == 0:
                logging.info("Epochs: {} : Train set training:, loss= {:.4f}".format(e, loss_buffer.item()))
                model.eval()

                labels_val = labels[val_idx]
                labels_test = labels[test_idx]

                # Full graph
                output = model(data_all)
                if isinstance(output, tuple):
                    output = output[0]

                loss_val=F.nll_loss(output[val_idx], labels_val)
                loss_test = F.nll_loss(output[test_idx], labels_test)

                acc_val = utils.accuracy(output[val_idx], labels_val)
                acc_test = utils.accuracy(output[test_idx], labels_test)

                writer.add_scalar('val_set_loss_curve', loss_val.item(), e)
                writer.add_scalar('val_set_acc_curve', acc_val.item(), e)

                writer.add_scalar('test_set_loss_curve', loss_test.item(), e)
                writer.add_scalar('test_set_acc_curve', acc_test.item(), e)

                if acc_val > best_val_acc:
                    best_val_acc = acc_val
                    best_test_acc = acc_test
                    best_it = e

            # 保存参数轨迹（每隔 param_save_interval 次）
            if e % args.param_save_interval == 0 and e>1:
                timestamps.append([p.detach().cpu() for p in model.parameters()])
                p_current = timestamps[-1]
                p_0 = timestamps[0]
                target_params = torch.cat([p_c.data.reshape(-1) for p_c in p_current], 0)
                starting_params = torch.cat([p0.data.reshape(-1) for p0 in p_0], 0)
                param_dist1 = torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
                writer.add_scalar('param_change', param_dist1.item(), e)
                logging.info(
                    '==============================={}-th iter with length of {}-th tsp'.format(e, len(timestamps)))

        logging.info("Valid set best results: accuracy= {:.4f}".format(best_val_acc.item()))
        logging.info("Test set best results: accuracy= {:.4f} within best iteration = {}".format(best_test_acc.item(),best_it))
        # print("Test set best results: accuracy= {:.4f} within best iteration = {}".format(best_test_acc.item(),best_it))
        # 添加当前模型的所有快照轨迹
        trajectories.append(timestamps)

        # 达到保存频率，保存轨迹文件
        if len(trajectories) == args.traj_save_interval:
            n = 0
            while os.path.exists(os.path.join(log_dir, f"replay_buffer_{n}.pt")):
                n += 1
            logging.info("Saving {}".format(os.path.join(log_dir, f"replay_buffer_{n}.pt")))
            if args.save_trajectories:
                torch.save(trajectories, os.path.join(log_dir, f"replay_buffer_{n}.pt"))
            trajectories = []

if __name__ == '__main__':
    parser = load_hnn_parser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--data_dir", type=str, default="data/pyg_data/hypergraph_dataset_updated", help="Data directory")
    parser.add_argument('--teacher_epochs', type=int, default=1000, help='training epochs')
    parser.add_argument('--teacher_nlayers', type=int, default=2)
    parser.add_argument('--teacher_hidden', type=int, default=256)
    parser.add_argument('--teacher_dropout', type=float, default=0.0)
    parser.add_argument('--lr_teacher', type=float, default=0.0005, help='initialization for buffer learning rate')
    parser.add_argument('--wd_teacher', type=float, default=0) # 权重
    parser.add_argument('--mom_teacher', type=float, default=0)
    parser.add_argument('--seed_teacher', type=int, default=15, help='Random seed.')
    parser.add_argument('--num_experts', type=int, default=10, help='training iterations')
    parser.add_argument('--param_save_interval', type=int, default=10)
    parser.add_argument('--traj_save_interval', type=int, default=10)
    parser.add_argument('--save_log', type=str, default='logs_WOCL', help='path to save logs')
    parser.add_argument('--save_trajectories',type=float,default=True, help='whether to save trajectories')
    parser.add_argument('--optim', type=str, default='SGD', choices=['Adam', 'SGD'], help='Default buffer_model type')
    parser.add_argument('--decay', type=int, default=0, choices=[1, 0], help='whether to decay lr at 1/2 training epochs')
    parser.add_argument('--decay_factor', type=float, default=0.1, help='decay factor of lr at 1/2 training epochs')
    parser.add_argument('--difficulty_type', type=str, default='random', choices=['node', 'hyperedge', 'random'])

    # GEOM
    parser.add_argument('--lam', type=float, default=0.70)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--scheduler', type=str, default='root')

    args = parser.parse_args()

    # 创建日志目录
    log_dir = './' + args.save_log + '/Buffer/{}-buffer'.format(args.dname)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('This is the log_dir: {}'.format(log_dir))
    # 创建 TensorBoard 日志目录
    writer = SummaryWriter(log_dir + '/tbx_log')
    # 开始训练
    main(args)
    logging.info(args)
    logging.info('Finish!, Log_dir: {}'.format(log_dir))
