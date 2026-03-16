import torch
import torch.nn as nn

import dhg
from dhg.nn import HyperGCNConv


class HyperGCN(nn.Module):
    r"""The HyperGCN convolution layer proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs paper`.

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        num_layers: int = 2,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.inlinear = nn.Linear(in_channels, hid_channels)
        self.outlinear = nn.Linear(hid_channels, num_classes)
        
        torch.nn.init.xavier_uniform_(self.inlinear.weight)
        torch.nn.init.xavier_uniform_(self.outlinear.weight)

        self.layers = nn.ModuleList([HyperGCNConv(in_channels=hid_channels,
                                                  out_channels=hid_channels,
                                                  drop_rate=drop_rate) for _ in range(num_layers)])
        self.act = nn.LeakyReLU()

    def reset_parameters(self):
        """
        初始化模型所有可训练参数，包括线性层和每个 HyperGCNConv 层
        """
        # 重置输入和输出线性层参数
        self.inlinear.reset_parameters()
        self.outlinear.reset_parameters()
        # 重置所有 HyperGCNConv 层参数
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph", return_emb=False) -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        X = self.inlinear(X)

        for layer in self.layers:
            X = layer(X, hg)
        
        if return_emb:
            return self.outlinear(X), X, None
        else:
            return self.outlinear(X), None, None
