import torch
import pickle
import random
import scipy as sp
from preprocessing import *
import numpy as np
import torch_sparse
from dhg import Hypergraph
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
from utils import filter_hyperedge
from convert_datasets_to_pygDataset import dataset_Hypergraph
from torch_geometric.data import Data
from torch_sparse import coalesce


def mask_split(labels, num_v):
    """
    对数据进行训练集、验证集和测试集的划分。
    采用StratifiedShuffleSplit方法，根据标签进行分层切分。
    """
    label_np = labels
    # 训练集、验证集、测试集划分为40%、20%、40%
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_val_idx, test_idx = next(sss.split(label_np, label_np))
    train_val_label = label_np[train_val_idx]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1/3, random_state=42)
    train_idx, val_idx = next(sss.split(train_val_label, train_val_label))
    # 初始化训练、验证和测试的掩码
    train_mask = torch.zeros(num_v, dtype=torch.bool)
    val_mask = torch.zeros(num_v, dtype=torch.bool)
    test_mask = torch.zeros(num_v, dtype=torch.bool)
    # 将对应的索引位置设置为True，标记为训练集、验证集和测试集
    train_mask[train_val_idx[train_idx]] = True
    val_mask[train_val_idx[val_idx]] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def load_data(args):
    """
    加载数据并进行预处理。
    根据不同的dataset name，选择对应的数据处理方式。
    """
    ### Load and preprocess data ###
    existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
                        'NTU2012', 'Mushroom',
                        'coauthor_cora', 'coauthor_dblp',
                        'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
                        'walmart-trips-100', 'house-committees-100',
                        'cora', 'citeseer', 'pubmed','tencent_2k']
    
    synthetic_list = ['amazon-reviews', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100']
    
    if args.dname in existing_dataset:
        dname = args.dname
        f_noise = args.feature_noise
        # 如果数据集是合成数据集且指定了噪声，加载带噪声的数据
        if (f_noise is not None) and dname in synthetic_list:
            p2raw = './data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname, 
                    feature_noise=f_noise,
                    p2raw = p2raw)
        else:
            # 根据不同数据集路径设置相应的raw数据路径
            if dname in ['cora', 'citeseer','pubmed']:
                p2raw = './data/AllSet_all_raw_data/cocitation/'
            elif dname in ['coauthor_cora', 'coauthor_dblp']:
                p2raw = './data/AllSet_all_raw_data/coauthorship/'
            elif dname in ['yelp']:
                p2raw = './data/AllSet_all_raw_data/yelp/'
            else:
                p2raw = './data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname,root = './data/pyg_data/hypergraph_dataset_updated/',
                                         p2raw = p2raw)
        # 获取数据
        data = dataset.data
        # 针对 Citeseer 数据集，剔除孤立节点并重编号
        if args.dname == 'citeseer':
            data = remove_isolated_nodes(data)
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        # 对某些数据集的标签进行处理，使得标签的最小值从 0 开始
        if args.dname in ['yelp', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100']:
            #         Shift the y label to start with 0
            args.num_classes = len(data.y.unique())
            data.y = data.y - data.y.min()
        # 如果数据中没有n_x属性，设置它
        if not hasattr(data, 'n_x'):
            data.n_x = torch.tensor([data.x.shape[0]])
        # 设置数据集中的超边数
        if not hasattr(data, 'num_hyperedges'):
            # note that we assume the he_id is consecutive.
            data.num_hyperedges = torch.tensor(
                [data.edge_index[0].max()-data.n_x[0]+1])

    # 将数据适配到模型中
    data = load_data_adapt_to_model(args, data)

    # ipdb.set_trace()
    #     Preprocessing
    # if args.method in ['SetGNN', 'SetGPRGNN', 'SetGNN-DeepSet']:
    
    #     Get splits
    # 获取训练、验证、测试的切分索引
    split_idx_lst = []
    for run in range(args.runs):
        split_idx = rand_train_test_idx(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
        split_idx_lst.append(split_idx)
    
    return data, split_idx_lst


def load_data_adapt_to_model(args, data):
    """
    根据不同的模型方法进行数据的适配和预处理。
    """
    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        if args.exclude_self:
            data = expand_edge_index(data)
    
        #     Compute deg normalization: option in ['all_one','deg_half_sym'] (use args.normtype)
        # data.norm = torch.ones_like(data.edge_index[0])
        data = norm_contruction(data, option=args.normtype)
    elif args.method in ['CEGCN', 'CEGAT']:
        data = ExtractV2E(data)
        data = ConstructV2V(data)
        data = norm_contruction(data, TYPE='V2V')
    
    elif args.method in ['HyperGCN']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        # 保存原始V2E边索引，用于排序
        data.V2E_edge_index = data.edge_index.clone()
        # 最小化超边索引，确保从0开始
        data.edge_index[1] -= data.edge_index[1].min()
    
    elif args.method in ['HNHN']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        H = ConstructH_HNHN(data)
        data = generate_norm_HNHN(H, data, args)
        data.edge_index[1] -= data.edge_index[1].min()
    
    elif args.method in ['HCHA', 'HGNN']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
     #    Make the first he_id to be 0
        data.edge_index[1] -= data.edge_index[1].min()
        # Save original V2E edge_index for sampling
        data.V2E_edge_index = data.edge_index.clone()
        # Build propagation matrix G for HGNN/HCHA
        data = ConstructH(data)
        data = generate_G_from_H(data)
    
    elif args.method in ['UniGCNII']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        data = ConstructH(data)
        data.edge_index = sp.csr_matrix(data.edge_index)
        # Compute degV and degE
        if args.cuda in [0,1]:
            device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        (row, col), value = torch_sparse.from_scipy(data.edge_index)
        V, E = row, col
        V, E = V.to(device), E.to(device)

        degV = torch.from_numpy(data.edge_index.sum(1)).view(-1, 1).float().to(device)
        from torch_scatter import scatter
        degE = scatter(degV[V], E, dim=0, reduce='mean')
        degE = degE.pow(-0.5)
        degV = degV.pow(-0.5)
        degV[torch.isinf(degV)] = 1
        args.UniGNN_degV = degV
        args.UniGNN_degE = degE
    
        V, E = V.cpu(), E.cpu()
        del V
        del E
    return data

def remove_isolated_nodes(data):
    """
    移除未参与任何超边的孤立节点，并重编号节点和超边
    """
    import numpy as _np
    device = data.x.device
    # 原始节点数
    if hasattr(data, 'n_x') and torch.is_tensor(data.n_x):
        orig_n_x = int(data.n_x.item())
    elif hasattr(data, 'n_x'):
        orig_n_x = int(data.n_x)
    else:
        orig_n_x = data.x.size(0)
    # 提取原始边索引
    edge_index = data.edge_index
    src = edge_index[0]
    dst = edge_index[1]
    # 找出所有有连接的节点ID
    node_mask = (src < orig_n_x) | (dst < orig_n_x)
    node_ids = torch.unique(torch.cat([src[node_mask][src[node_mask] < orig_n_x], dst[node_mask][dst[node_mask] < orig_n_x]]))
    node_ids_list = node_ids.cpu().numpy().tolist()
    new_n_x = len(node_ids_list)
    # 构建节点映射 old->new
    node_old2new = {old: new for new, old in enumerate(node_ids_list)}
    # 找出所有有连接的超边ID
    he_mask = (src >= orig_n_x) | (dst >= orig_n_x)
    he_ids = torch.unique(torch.cat([src[he_mask][src[he_mask] >= orig_n_x], dst[he_mask][dst[he_mask] >= orig_n_x]]))
    he_ids_list = he_ids.cpu().numpy().tolist()
    new_num_he = len(he_ids_list)
    # 构建超边映射 old_he_id->new_he_id
    he_old2new = {old: new_n_x + i for i, old in enumerate(he_ids_list)}
    # 重建 edge_index 列表
    u_list, v_list = [], []
    for u_old, v_old in zip(src.cpu().numpy(), dst.cpu().numpy()):
        if u_old < orig_n_x:
            u_new = node_old2new[u_old]
        else:
            u_new = he_old2new.get(u_old, None)
            if u_new is None:
                continue
        if v_old < orig_n_x:
            v_new = node_old2new[v_old]
        else:
            v_new = he_old2new.get(v_old, None)
            if v_new is None:
                continue
        u_list.append(u_new)
        v_list.append(v_new)
    new_edge_index = torch.tensor([u_list, v_list], dtype=torch.long, device=device)
    data.edge_index = new_edge_index
    # 更新节点特征和标签
    data.x = data.x[node_ids_list]
    data.y = data.y[node_ids_list]
    data.n_x = torch.tensor([new_n_x], device=device)
    data.num_hyperedges = torch.tensor([new_num_he], device=device)
    return data

