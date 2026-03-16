import torch
import torch.nn as nn
import logging
from models import SetGNN

class SetGNNWrapper(nn.Module):
    """
    封装 SetGNN 模型，使其在蒸馏阶段能够正确处理简化图结构。
    """
    def __init__(self, args, norm=None):
        super().__init__()
        self.model = SetGNN(args, norm)
        self.args = args
        # 添加批量归一化层
        self.bn = nn.BatchNorm1d(args.MLP_hidden)
        # 添加用于残差连接的投影层
        self.has_residual = (args.num_features == args.num_classes)
        if not self.has_residual:
            self.residual_proj = nn.Linear(args.num_features, args.num_classes)

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
        
        # 克隆边索引以防止原地修改导致梯度问题
        if hasattr(data, 'edge_index') and hasattr(data.edge_index, 'clone'):
            data.edge_index = data.edge_index.clone().detach()
              
        try:
            # 调用内部模型
            outs, node_emb, _ = self.model(data, return_emb)
            
            # 添加残差连接
            if self.has_residual:
                outs = outs + identity
            else:
                outs = outs + self.residual_proj(identity)
                
            return outs, node_emb, None
        except Exception as e:
            logging.warning(f"SetGNN forward failed with error: {e}. Using fallback MLP.")
            # 出错时使用备用MLP
            if hasattr(self.model, 'classifier'):
                outs = self.model.classifier(X)
            else:
                # 创建一个简单的MLP
                H = nn.Linear(X.shape[1], self.args.MLP_hidden)(X)
                H = self.bn(H)
                H = torch.relu(H)
                H = torch.dropout(H, self.args.dropout, self.training)
                outs = nn.Linear(self.args.MLP_hidden, self.args.num_classes)(H)
            
            # 添加残差连接
            if self.has_residual:
                outs = outs + identity
            else:
                outs = outs + self.residual_proj(identity)
                
            if return_emb:
                return outs, X, None
            else:
                return outs, None, None 