import torch
import torch.nn as nn
import torch.nn.functional as F
from dhg.nn import HGNNPConv


class HGNN_PLUS(nn.Module):
    r"""The HGNN + convolution layer proposed in ``HGNN+: General Hypergraph Neural Networks'' paper (IEEE TPAMI 2022).

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

        self.inlinear = nn.Linear(in_channels, hid_channels)
        self.outlinear = nn.Linear(hid_channels, num_classes)
        
        torch.nn.init.xavier_uniform_(self.inlinear.weight)
        torch.nn.init.xavier_uniform_(self.outlinear.weight)

        self.layers = nn.ModuleList([HGNNPConv(in_channels=hid_channels,
                                               out_channels=hid_channels,
                                               drop_rate=drop_rate) for _ in range(num_layers)])
        self.act = nn.LeakyReLU()

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
