import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
import datetime
import json
import argparse
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from meta_gradient import MetaGtt
from utils.dataloader import load_data
from model_data_parser import load_hnn_parser
from utils.utils import str2bool

def main():
    parser = load_hnn_parser()
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")
    parser.add_argument("--wandb",type=int,default=0,help="Use wandb")
    parser.add_argument("--whole_data", type=int, default=0)
    parser.add_argument("--transductive", type=int, default=1, help="Transductive setting")
    parser.add_argument("--data_dir", type=str, default="data/pyg_data/hypergraph_dataset_updated", help="Data directory")
    parser.add_argument("--save_dir", type=str, default="save_H", help="Save directory")
    parser.add_argument("--keep_ratio", type=float, default=1.0)
    parser.add_argument("--reduction_rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=15, help="Random seed")
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--save", type=int, default=1)
    parser.add_argument("--early_patience", type=int, default=10)
    parser.add_argument("--outer_patience", type=int, default=20)
    # SFGC coreset init parameters
    parser.add_argument("--coreset_init_path", type=str, default="logs_AllDeepSets/Coreset")
    parser.add_argument("--coreset_log", type=str, default="logs_AllDeepSets")
    parser.add_argument("--save_log", type=str, default="logs_AllDeepSets", help="path to save logs")
    parser.add_argument("--core_method", type=str, default="herding", choices=["kcenter", "herding", "random"])
    parser.add_argument("--coreset_hidden", type=float, default=256)
    parser.add_argument("--coreset_seed", type=int, default=15)
    parser.add_argument("--normalize_features", type=bool, default=True)
    parser.add_argument("--coreset_init_weight_decay", type=float, default=5e-4)
    parser.add_argument("--coreset_init_lr", type=float, default=0.01)
    parser.add_argument("--lr_coreset", type=float, default=0.01)
    parser.add_argument("--wd_coreset", type=float, default=5e-4)
    parser.add_argument("--coreset_nlayers", type=int, default=2, help="Random seed.")
    parser.add_argument("--coreset_epochs", type=int, default=1)
    parser.add_argument("--coreset_save", type=int, default=1)
    parser.add_argument("--coreset_load_npy", type=str, default="")
    parser.add_argument("--coreset_opt_type_train", type=str, default="Adam")
    parser.add_argument("--coreset_runs",type=int,default=10)
    parser.add_argument("--config", type=str, default="config_geom.json", help="Path to the config JSON file")
    parser.add_argument("--section", type=str, default="cora-r05", help="the experiments needs to run",)
    parser.add_argument("--ITER", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--samp_iter", type=int, default=1)
    parser.add_argument("--samp_num_per_class", type=int, default=1)
    parser.add_argument("--ntk_reg",type=float,default=1.0)
    parser.add_argument("--syn_steps", type=int, default=200)
    parser.add_argument("--expert_epochs",type=int,default=400)
    parser.add_argument("--start_epoch", type=int, default=10)
    parser.add_argument("--lr_feat", type=float, default=0.01)
    parser.add_argument("--lr_lr",type=float,default=0.01)
    parser.add_argument("--lr_student", type=float, default=0.0001)
    parser.add_argument("--student_nlayers", type=int, default=2)
    parser.add_argument("--student_hidden", type=int, default=256)
    parser.add_argument("--student_dropout", type=float, default=0.0)
    parser.add_argument("--load_all", action="store_true",
        help="only use if you can fit all expert trajectories into RAM",)
    parser.add_argument("--max_files", type=int, default=None,
        help="number of expert files to read (leave as None unless doing ablations)",)
    parser.add_argument("--max_experts", type=int, default=None,
        help="number of experts to read per file (leave as None unless doing ablations)",)
    parser.add_argument("--eval_type", type=str, default="S", help="eval_mode, check utils.py for more info",)
    parser.add_argument("--initial_save", type=int, default=0, help="whether save initial feat and syn")
    parser.add_argument("--interval_buffer", type=int, default=1, choices=[0, 1], help="whether use interval buffer",)
    parser.add_argument("--rand_start", type=int, default=1, choices=[0, 1], help="whether use random start",)
    parser.add_argument("--optimizer_con", type=str, default="Adam", help="See choices", choices=["Adam", "SGD"],)
    parser.add_argument("--optim_lr", type=int, default=0, help="whether use LR lr learning optimizer")
    parser.add_argument("--optimizer_lr", type=str, default="Adam", help="See choices", choices=["Adam", "SGD"],)
    parser.add_argument("--lr_decay", type=int, default=0,  # 设置默认值为 0，表示没有学习率衰减
        help="Whether to apply learning rate decay")
    parser.add_argument('--best_ntk_score', type=int, default=1)    
    # GEOM  
    parser.add_argument('--max_start_epoch_s', type=int, default=10) 
    parser.add_argument('--lam', default=0.70)
    parser.add_argument('--T', default=500)
    parser.add_argument('--scheduler', default='root')
    parser.add_argument('--beta', type=float, default=0.1)  # coefficient for loss_clom
    parser.add_argument("--soft_label", type=str2bool, default=False)
    parser.add_argument('--lr_y', type=float, default=0.01) # optimizer for label learning
    parser.add_argument('--lr_tem', type=float, default=0.01)
    parser.add_argument('--maxtem', type=float, default=0.0)
    parser.add_argument('--tem', type=float, default=0.0)
    parser.add_argument('--max_start_epoch', type=int, default=500)
    parser.add_argument('--min_start_epoch', type=int, default=5)
    parser.add_argument('--test_lr_model', type=float,default=0.01)  
    parser.add_argument('--test_wd', type=float, default=5e-4)
    parser.add_argument('--test_model_iters', type=int, default=100)
    parser.add_argument('--nruns', type=int, default=10)    
    parser.add_argument('--test_dropout', type=float, default=0.0)
    parser.add_argument('--test_opt_type', type=str, default='Adam')
    parser.add_argument('--test_nlayers', type=int, default=2)
    parser.add_argument('--test_hidden', type=int, default=256)
    parser.add_argument('--expanding_window', type=bool, default=True, help='new matching strategy')

    
    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(True)

    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 设置日志目录
    log_dir = './' + args.save_log + '/Distill/{}-reduce_{}-{}'.format(args.dname, str(args.reduction_rate),
                                                            datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 设置缓冲区目录
    buffer_path = os.path.join(args.save_log, "Buffer/{}-buffer".format(args.dname))
    if not os.path.exists(buffer_path):
        os.makedirs(buffer_path)

    args.__setattr__("buffer_path", buffer_path)
    args.__setattr__("device","cuda:{}".format(args.gpu_id))
    args.__setattr__("log_dir", log_dir)

    # 设置日志格式
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('This is the log_dir: {}'.format(log_dir))

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logging.info("args = {}".format(args))

    # 设置数据目录
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # 设置保存目录
    if not os.path.exists(f"{args.save_dir}"):
        os.makedirs(f"{args.save_dir}")

    data, split_idx_lst = load_data(args)

    # 创建MetaGtt实例
    agent = MetaGtt(data, split_idx_lst, args, device=device)
      
    # 创建SummaryWriter实例
    writer = SummaryWriter()

    # 执行蒸馏过程
    agent.distill(writer)
    

if __name__ == "__main__":
    main()
