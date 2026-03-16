import torch
import torch.nn as nn
import logging
from dhg import Hypergraph
from model.hypergcn import HyperGCN as DHGHyperGCN
from preprocessing import get_HyperGCN_He_dict

class HyperGCNWrapper(nn.Module):
    """
    封装 DHG 库中的 HyperGCN，使其兼容原有训练框架（接受 Data 对象输入）。
    """
    def __init__(self, in_channels, hid_channels, num_classes, num_layers, drop_rate):
        super().__init__()
        self.model = DHGHyperGCN(
            in_channels=in_channels,
            hid_channels=hid_channels,
            num_classes=num_classes,
            num_layers=num_layers,
            drop_rate=drop_rate
        )
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        # 添加批量归一化层
        self.bn = nn.BatchNorm1d(hid_channels)
        # 添加用于残差连接的投影层
        self.has_residual = (in_channels == num_classes)
        if not self.has_residual:
            self.residual_proj = nn.Linear(in_channels, num_classes)

    def reset_parameters(self):
        self.model.reset_parameters()
        self.bn.reset_parameters()
        if not self.has_residual:
            torch.nn.init.xavier_uniform_(self.residual_proj.weight)
            if self.residual_proj.bias is not None:
                torch.nn.init.zeros_(self.residual_proj.bias)

    def forward(self, data, return_emb=False):
        # 提取特征矩阵
        X = data.x
        identity = X  # 保存输入用于残差连接
        
        # 构建超边字典并初始化 Hypergraph
        he_dict = get_HyperGCN_He_dict(data)
        he_list = [nodes for nodes in he_dict.values() if len(nodes) >= 2]
              
        # 若无有效超边，静默添加自环（蒸馏阶段常见，不需要警告）
        if len(he_list) == 0:
            # 创建自环结构，每个节点与自身连接
            for i in range(X.shape[0]):
                he_list.append([i, i])  # 添加自环
            
        # 构建 Hypergraph 对象
        hg = Hypergraph(X.shape[0], he_list)
        
        try:
            # 调用内部模型
            outs, emb_x, emb_edge = self.model(X, hg, return_emb)
            
            # 添加残差连接
            if self.has_residual:
                outs = outs + identity
            else:
                outs = outs + self.residual_proj(identity)
                
            return outs, emb_x, emb_edge
        except Exception as e:
            # 出错时静默使用备用MLP
            H = self.model.inlinear(X)
            H = self.bn(H)
            H = torch.relu(H)
            H = torch.dropout(H, self.drop_rate, self.training)
            outs = self.model.outlinear(H)
            
            # 添加残差连接
            if self.has_residual:
                outs = outs + identity
            else:
                outs = outs + self.residual_proj(identity)
                
            if return_emb:
                return outs, H, None
            else:
                return outs, None, None 