import os
import sys
import math
import logging
import datetime
import torch.nn as nn
import argparse
import torch
import torch.nn.functional as F
import numpy as np


def init_logger(args, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('This is the log_dir: {}'.format(log_dir))
    logging.info(args)
    logging.info('Finish!, Log_dir: {}'.format(log_dir))


def init_wandb(args, tags=None):
    import wandb
    wandb.init(project='Condense_HGraph', entity='', config=vars(args), tags=[f'{tag}' for tag in tags])
    args = argparse.Namespace(**wandb.config)
    return args


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


def get_summary_writer(*args, **kwargs):
    try:
        from tensorboardX import SummaryWriter
        return SummaryWriter(*args, **kwargs)
    except ImportError:
        try:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter(*args, **kwargs)
        except ImportError:
            logging.warning("TensorBoard is not installed; using a no-op SummaryWriter.")
            return NullSummaryWriter()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)


def resolve_device(device=None, gpu_id=None):
    requested = str(device or "auto").lower()
    if requested in {"auto", "default"}:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{0 if gpu_id is None else gpu_id}")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        logging.warning("MPS was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _init_weights(module):
    if isinstance(module, nn.LayerNorm) or isinstance(module, nn.BatchNorm1d):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'weight') and module.weight is not None:
        # nn.init.xavier_uniform_(module.weight)
        stdv = 1. / math.sqrt(module.weight.T.size(1))
        module.weight.data.uniform_(-stdv, stdv)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.uniform_(-stdv, stdv)


def init_parameters(model):
    model.apply(_init_weights)


def to_device(device, *args):
    return [x.to(device) for x in args]


def filter_hyperedge(hyperedges, node_ids):
    sorted_node_ids = sorted(node_ids)
    node_id2idx = {node_id: idx for idx, node_id in enumerate(sorted_node_ids)}
    new_hyperedges = []
    for hyperedge in hyperedges:
        new_hyperedge = [node_id2idx[node_id] for node_id in hyperedge if node_id in node_id2idx]
        if len(new_hyperedge) > 1:
            new_hyperedges.append(tuple(new_hyperedge))
    logging.info(f'Filter hyperedges: {len(hyperedges)} -> {len(new_hyperedges)}')
    return new_hyperedges


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = [model, 'MLP']
    elif eval_mode == 'S': # itself
        model_eval_pool = [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def sort_training_nodes_bipartite(edge_index, labels, num_nodes, train_idx, device='cuda'):
    difficulty = node_difficulty_bipartite(edge_index, labels, num_nodes, train_idx, device=device)
    _, indices = torch.sort(difficulty)
    indices = indices.cpu().numpy()
    sorted_trainset = train_idx[indices]
    return sorted_trainset


def node_difficulty_bipartite(edge_index, labels, num_nodes, train_idx, device='cuda'):
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    idx_node = edge_index[0].to(device)
    idx_edge = edge_index[1].to(device)
    num_edges = int(idx_edge.max().item()) + 1
    num_classes = int(labels.max().item()) + 1

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    valid_node_mask = (idx_node >= 0) & (idx_node < num_nodes)
    labeled_link_mask = torch.zeros_like(valid_node_mask, dtype=torch.bool, device=device)
    labeled_link_mask[valid_node_mask] = train_mask[idx_node[valid_node_mask]]

    # 每个超边中的类别分布。Only labeled training nodes are used so PaS
    # cannot access validation/test labels when estimating structure-label consistency.
    labeled_nodes = idx_node[labeled_link_mask]
    labeled_edges = idx_edge[labeled_link_mask]
    one_hot = F.one_hot(labels[labeled_nodes], num_classes=num_classes).float()
    label_sum_per_edge = torch.zeros((num_edges, num_classes), device=device).index_add_(
        0, labeled_edges, one_hot
    )
    edge_size = torch.zeros((num_edges, 1), device=device).index_add_(
        0, labeled_edges, torch.ones_like(labeled_edges, dtype=torch.float).unsqueeze(1)
    )
    edge_distribution = label_sum_per_edge / (edge_size + 1e-10)

    # 每个节点平均参与的超边分布
    edge_label_for_nodes = edge_distribution[idx_edge]  # shape: (num_links, num_classes)
    node_distribution_sum = torch.zeros((num_nodes, num_classes), device=device).index_add_(
        0, idx_node, edge_label_for_nodes
    )
    edge_count_per_node = torch.zeros((num_nodes, 1), device=device).index_add_(
        0, idx_node, torch.ones_like(idx_node, dtype=torch.float).unsqueeze(1)
    )
    node_label_distribution = node_distribution_sum / (edge_count_per_node + 1e-10)

    entropy = -torch.sum(
        node_label_distribution * torch.log(node_label_distribution + torch.exp(torch.tensor(-20.0, device=device))),
        dim=1
    )

    return entropy[train_idx]

def sort_nodes_by_hyperedge_difficulty(edge_index, labels, num_nodes, train_idx, device='cuda'):
    edge_difficulty = hyperedge_difficulty_bipartite(edge_index, labels, num_nodes, train_idx, device=device)
    node_difficulty = node_difficulty_from_edge(edge_difficulty, edge_index, num_nodes, device=device)
    _, sorted_node_indices = torch.sort(node_difficulty[train_idx])
    return train_idx.to(sorted_node_indices.device)[sorted_node_indices].cpu().numpy()



def hyperedge_difficulty_bipartite(edge_index, labels, num_nodes, train_idx, device='cuda'):
    idx_node = edge_index[0].to(device)
    idx_edge = edge_index[1].to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)

    num_edges = int(idx_edge.max().item()) + 1
    num_classes = int(labels.max().item()) + 1

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    valid_node_mask = (idx_node >= 0) & (idx_node < num_nodes)
    labeled_link_mask = torch.zeros_like(valid_node_mask, dtype=torch.bool, device=device)
    labeled_link_mask[valid_node_mask] = train_mask[idx_node[valid_node_mask]]

    labeled_nodes = idx_node[labeled_link_mask]
    labeled_edges = idx_edge[labeled_link_mask]
    one_hot = F.one_hot(labels[labeled_nodes], num_classes=num_classes).float()
    label_sum_per_edge = torch.zeros((num_edges, num_classes), device=device).index_add_(
        0, labeled_edges, one_hot
    )
    edge_size = torch.zeros((num_edges, 1), device=device).index_add_(
        0, labeled_edges, torch.ones_like(labeled_edges, dtype=torch.float).unsqueeze(1)
    )
    edge_distribution = label_sum_per_edge / (edge_size + 1e-10)
    entropy = -torch.sum(edge_distribution * torch.log(edge_distribution + torch.exp(torch.tensor(-20.0, device=device))), dim=1)
    return entropy


def node_difficulty_from_edge(edge_entropy, edge_index, num_nodes, device='cuda'):
    idx_node = edge_index[0].to(device)
    idx_edge = edge_index[1].to(device)
    edge_entropy = edge_entropy.to(device)

    node_entropy_sum = torch.zeros((num_nodes,), device=device).index_add_(
        0, idx_node, edge_entropy[idx_edge]
    )
    node_edge_count = torch.zeros((num_nodes,), device=device).index_add_(
        0, idx_node, torch.ones_like(idx_node, dtype=torch.float)
    )
    node_difficulty = node_entropy_sum / (node_edge_count + 1e-10)
    return node_difficulty


def training_scheduler(lam, t, T, scheduler="geom"):
    """
    训练调度器
    
    Args:
        lam: 初始值，通常在 (0, 1) 之间
        t: 当前步数
        T: 总步数
        scheduler: 调度器类型
            linear: 线性调度，增长速度恒定
            root: 平方根调度，前期增长慢，后期加速
            geom: 几何调度（指数型），前期几乎不变，后期快速接近1
        
    Returns:
        float: 调度值（返回一个随训练进度逐渐增长的浮点数，最大不超过 1）
    """
    if scheduler == "linear":
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == "root":
        return min(1, math.sqrt(lam**2 + (1 - lam**2) * t / T))
    elif scheduler == "geom":
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))
    

class HyperGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, reapproximate=False, cuda=False):
        super(HyperGraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reapproximate = reapproximate
        self.cuda = cuda
        
        # 定义可学习的权重矩阵
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        
        # 添加注意力机制
        self.attention = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )
        
        # 添加层归一化
        self.layer_norm = nn.LayerNorm(out_channels)
        
        # 添加dropout
        self.dropout = nn.Dropout(0.1)
        
        # 初始化参数
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化模型参数"""
        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.bias)
        
    def forward(self, structure, x, mediators=None):
        """
        前向传播函数
        """
        # 获取输入张量的设备
        device = x.device
        
        # 特征归一化
        x = F.normalize(x, p=2, dim=1)
        
        if self.reapproximate:
            # 如果使用重新近似，需要构建拉普拉斯矩阵
            L = self._construct_laplacian(structure, mediators, x.size(0))
            # 确保L在正确的设备上
            L = L.to(device)
            # 使用拉普拉斯矩阵进行消息传递
            out = torch.matmul(L, x)
        else:
            # 使用注意力机制进行消息传递
            out = self._attention_message_passing(structure, x)
            
        # 应用dropout
        out = self.dropout(out)
            
        # 应用线性变换
        out = torch.matmul(out, self.weight.to(device)) + self.bias.to(device)
        
        # 应用层归一化
        out = self.layer_norm(out)
        
        return out
        
    def _attention_message_passing(self, structure, x):
        """
        基于注意力机制的消息传递
        """
        # 将列表转换为张量
        if isinstance(structure, list):
            structure = torch.tensor(structure, device=x.device)
            
        # 获取源节点和目标节点
        source_nodes = structure[0]
        target_nodes = structure[1]
        
        # 计算注意力分数
        source_features = x[source_nodes]
        target_features = x[target_nodes]
        attention_input = torch.cat([source_features, target_features], dim=1)
        attention_scores = self.attention(attention_input)
        attention_weights = F.softmax(attention_scores, dim=0)
        
        # 加权聚合邻居节点的特征
        weighted_features = source_features * attention_weights
        out = scatter_add(weighted_features, target_nodes, dim=0, dim_size=x.size(0))
        
        # 归一化
        degree = scatter_add(torch.ones_like(source_nodes), target_nodes, dim=0, dim_size=x.size(0))
        degree = torch.clamp(degree, min=1)
        out = out / degree.unsqueeze(-1)
        
        return out
        
    def _construct_laplacian(self, structure, mediators, num_nodes):
        """
        构建超图的拉普拉斯矩阵
        """
        # 获取输入张量的设备
        device = structure.device if isinstance(structure, torch.Tensor) else 'cuda' if self.cuda else 'cpu'
            
        # 将列表转换为张量
        if isinstance(structure, list):
            structure = torch.tensor(structure, device=device)
            
        # 获取边数
        if isinstance(structure, torch.Tensor):
            num_edges = structure[1].max().item() + 1
        else:
            num_edges = max(structure[1]) + 1
            
        # 构建关联矩阵
        H = torch.zeros((num_nodes, num_edges), device=device)
        for i in range(len(structure[0])):
            H[structure[0][i], structure[1][i]] = 1
            
        # 计算度矩阵
        D_v = torch.diag(H.sum(dim=1)).to(device)
        D_e = torch.diag(H.sum(dim=0)).to(device)
        
        # 处理可能的零度节点
        D_v = torch.where(D_v == 0, torch.ones_like(D_v), D_v)
        D_e = torch.where(D_e == 0, torch.ones_like(D_e), D_e)
        
        # 添加小的扰动以避免奇异矩阵
        epsilon = 1e-5
        D_v = D_v + epsilon * torch.eye(D_v.size(0), device=device)
        D_e = D_e + epsilon * torch.eye(D_e.size(0), device=device)
        
        # 计算归一化的拉普拉斯矩阵
        D_v_inv_sqrt = torch.pow(D_v, -0.5)
        D_e_inv = torch.inverse(D_e)
        L = torch.matmul(torch.matmul(D_v_inv_sqrt, H), D_e_inv)
        L = torch.matmul(L, torch.matmul(H.t(), D_v_inv_sqrt))
        
        # 确保拉普拉斯矩阵是对称的
        L = (L + L.t()) / 2
        
        return L.to(device)

def sort_nodes_by_H_matrix(H_matrix, labels, num_nodes, train_idx, device='cuda'):
    """
    基于H矩阵对节点进行排序，专门用于HGNN模型
    
    参数:
    H_matrix: 超图的关联矩阵，形状为 [num_nodes, num_hyperedges]
    labels: 节点标签
    num_nodes: 节点数量
    train_idx: 训练集索引
    device: 设备
    
    返回:
    sorted_trainset: 排序后的训练集索引
    """
    # 安全检查
    if H_matrix is None or H_matrix.shape[0] == 0 or H_matrix.shape[1] == 0:
        # 如果H矩阵为空，则直接返回随机排序的训练索引
        if isinstance(train_idx, torch.Tensor):
            sorted_trainset = train_idx.clone().cpu().numpy()
        else:
            sorted_trainset = train_idx.copy()
        np.random.shuffle(sorted_trainset)
        return sorted_trainset
    
    # 确保H矩阵是PyTorch张量并在正确的设备上
    if not isinstance(H_matrix, torch.Tensor):
        H_matrix = torch.tensor(H_matrix, device=device)
    else:
        H_matrix = H_matrix.to(device)
    
    # 计算每个节点的度（它连接到的超边数量）
    node_degrees = H_matrix.sum(dim=1)
    
    # 计算每个节点的标签熵
    # 首先获取训练集中每个节点所连接的超边
    train_idx_tensor = torch.tensor(train_idx, device=device)
    
    # 计算困难度指标：节点度 * 连接到的超边的标签多样性
    node_difficulty = torch.zeros(num_nodes, device=device)
    
    for i in range(num_nodes):
        # 跳过非训练集节点
        if i not in train_idx_tensor:
            continue
            
        # 获取当前节点连接的所有超边
        connected_hyperedges = torch.where(H_matrix[i] > 0)[0]
        
        # 如果节点没有连接的超边，则跳过
        if len(connected_hyperedges) == 0:
            continue
            
        # 获取这些超边连接的所有节点（除了当前节点）
        connected_nodes = []
        for he in connected_hyperedges:
            nodes = torch.where(H_matrix[:, he] > 0)[0]
            nodes = nodes[nodes != i]  # 排除当前节点
            connected_nodes.extend(nodes.tolist())
            
        # 如果没有连接的其他节点，则跳过
        if len(connected_nodes) == 0:
            continue
            
        # 获取这些节点的标签
        connected_nodes = torch.tensor(connected_nodes, device=device)
        connected_nodes = connected_nodes[connected_nodes < labels.shape[0]]  # 确保索引有效
        if len(connected_nodes) == 0:
            continue
            
        neighbor_labels = labels[connected_nodes]
        
        # 计算标签分布的熵
        unique_labels, counts = torch.unique(neighbor_labels, return_counts=True)
        probs = counts.float() / counts.sum()
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        
        # 困难度 = 度 * 熵
        node_difficulty[i] = node_degrees[i] * entropy
    
    # 对训练集中的节点按困难度排序
    train_difficulty = node_difficulty[train_idx_tensor]
    _, indices = torch.sort(train_difficulty)
    indices = indices.cpu().numpy()
    sorted_trainset = train_idx[indices]
    
    return sorted_trainset
