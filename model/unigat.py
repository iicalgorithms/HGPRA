import torch
import torch.nn as nn

import dhg
from dhg import Hypergraph


class UNIGAT(nn.Module):
    r"""The UniGAT convolution layer proposed in ``UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks'' paper (IJCAI 2021).

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

        self.layers = nn.ModuleList([UniGATConv(in_channels=hid_channels,
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
            X, Y = layer(X, hg)
        
        if return_emb:
            return self.outlinear(X), X, Y
        else:
            return self.outlinear(X), None, None


class UniGATConv(nn.Module):
    r"""The UniGAT convolution layer proposed in `UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks <https://arxiv.org/pdf/2105.00956.pdf>`_ paper (IJCAI 2021).

    Sparse Format:

    .. math::
        \left\{
            \begin{aligned}
                \alpha_{i e} &=\sigma\left(a^{T}\left[W h_{\{i\}} ; W h_{e}\right]\right) \\
                \tilde{\alpha}_{i e} &=\frac{\exp \left(\alpha_{i e}\right)}{\sum_{e^{\prime} \in \tilde{E}_{i}} \exp \left(\alpha_{i e^{\prime}}\right)} \\
                \tilde{x}_{i} &=\sum_{e \in \tilde{E}_{i}} \tilde{\alpha}_{i e} W h_{e}
            \end{aligned}
        \right. .
    
    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): The dropout probability. If ``dropout <= 0``, the layer will not drop values. Defaults to ``0.5``.
        ``atten_neg_slope`` (``float``): Hyper-parameter of the ``LeakyReLU`` activation of edge attention. Defaults to ``0.2``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        atten_neg_slope: float = 0.2,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.atten_dropout = nn.Dropout(drop_rate)
        self.atten_act = nn.LeakyReLU(atten_neg_slope)
        self.act = nn.ELU(inplace=True)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)
        self.atten_e = nn.Linear(out_channels, 1, bias=False)
        self.atten_dst = nn.Linear(out_channels, 1, bias=False)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        X = self.theta(X)
        Y = hg.v2e(X, aggr="mean")
        # ===============================================
        alpha_e = self.atten_e(Y)
        e_atten_score = alpha_e[hg.e2v_src]
        e_atten_score = self.atten_dropout(self.atten_act(e_atten_score).squeeze())
        # ================================================================================
        # We suggest to add a clamp on attention weight to avoid Nan error in training.
        e_atten_score = torch.clamp(e_atten_score, min=0.001, max=5)
        # ================================================================================
        X = hg.e2v(Y, aggr="softmax_then_sum", e2v_weight=e_atten_score)

        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
        return X, Y
