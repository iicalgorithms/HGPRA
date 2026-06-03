#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

"""
This script contains all models in our paper.
"""

import torch
import utils

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv
from utils.layers import *

import math 

from torch_scatter import scatter
from torch_geometric.utils import softmax

from utils.layers import HNHNConv, HGNN_conv, HypergraphConv, HalfNLHconv


#  This part is for HyperGCN

class HyperGCN(nn.Module):
    def __init__(self, V, E, X, num_features, num_layers, num_classses, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperGCN, self).__init__()
        d, l, c = num_features, num_layers, num_classses
        cuda = args.cuda  # and torch.cuda.is_available()

        h = [d]
        for i in range(l-1):
            power = l - i + 2
            if args.dname == 'citeseer':
                power = l - i + 4
            h.append(2**power)
        h.append(c)

        if args.HyperGCN_fast:
            reapproximate = False
            structure = utils.Laplacian(V, E, X, args.HyperGCN_mediators)
        else:
            reapproximate = True
            structure = E

        self.layers = nn.ModuleList([utils.HyperGraphConvolution(
            h[i], h[i+1], reapproximate, cuda) for i in range(l)])
        self.do, self.l = args.dropout, num_layers
        self.structure, self.m = structure, args.HyperGCN_mediators
        self.E = E

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data, return_emb=False):
        """
        an l-layer GCN
        """
        do, l, m = self.do, self.l, self.m
        H = data.x

        for i, hidden in enumerate(self.layers):
            V = F.relu(hidden(self.structure, H, m))
            if i < l - 1:
                H = F.dropout(V, do, training=self.training)

        if return_emb:
            return V, H, None
        else:
            return V, None, None


class CEGCN(MessagePassing):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 dropout,
                 Normalization='bn'
                 ):
        super(CEGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.normalizations = nn.ModuleList()

        if Normalization == 'bn':
            self.convs.append(GCNConv(in_dim, hid_dim, normalize=False))
            self.normalizations.append(nn.BatchNorm1d(hid_dim))
            for _ in range(num_layers-2):
                self.convs.append(GCNConv(hid_dim, hid_dim, normalize=False))
                self.normalizations.append(nn.BatchNorm1d(hid_dim))

            self.convs.append(GCNConv(hid_dim, out_dim, normalize=False))
        else:  # default no normalizations
            self.convs.append(GCNConv(in_dim, hid_dim, normalize=False))
            self.normalizations.append(nn.Identity())
            for _ in range(num_layers-2):
                self.convs.append(GCNConv(hid_dim, hid_dim, normalize=False))
                self.normalizations.append(nn.Identity())

            self.convs.append(GCNConv(hid_dim, out_dim, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        for normalization in self.normalizations:
            if normalization.__class__.__name__ != 'Identity':
                normalization.reset_parameters()

    def forward(self, data, return_emb=False):
        #         Assume edge_index is already V2V
        x, edge_index, norm = data.x, data.edge_index, data.norm
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, norm)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        outs = self.convs[-1](x, edge_index, norm)
        
        node_emb = x if return_emb else None
        return outs, node_emb, None


class CEGAT(MessagePassing):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 heads,
                 output_heads,
                 dropout,
                 Normalization='bn'
                 ):
        super(CEGAT, self).__init__()
        self.convs = nn.ModuleList()
        self.normalizations = nn.ModuleList()

        if Normalization == 'bn':
            self.convs.append(GATConv(in_dim, hid_dim, heads))
            self.normalizations.append(nn.BatchNorm1d(hid_dim))
            for _ in range(num_layers-2):
                self.convs.append(GATConv(heads*hid_dim, hid_dim))
                self.normalizations.append(nn.BatchNorm1d(hid_dim))

            self.convs.append(GATConv(heads*hid_dim, out_dim,
                                      heads=output_heads, concat=False))
        else:  # default no normalizations
            self.convs.append(GATConv(in_dim, hid_dim, heads))
            self.normalizations.append(nn.Identity())
            for _ in range(num_layers-2):
                self.convs.append(GATConv(hid_dim*heads, hid_dim))
                self.normalizations.append(nn.Identity())

            self.convs.append(GATConv(hid_dim*heads, out_dim,
                                      heads=output_heads, concat=False))

        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()
        for normalization in self.normalizations:
            if normalization.__class__.__name__ != 'Identity':
                normalization.reset_parameters()

    def forward(self, data, return_emb=False):
        #         Assume edge_index is already V2V
        x, edge_index, norm = data.x, data.edge_index, data.norm
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        outs = self.convs[-1](x, edge_index)
        node_emb = x if return_emb else None
        return outs, node_emb, None


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

        # 添加批量归一化
        self.bn1 = nn.BatchNorm1d(n_hid)
        
        # 添加用于残差连接的投影层（如果输入和输出维度不同）
        self.has_residual = (in_ch == n_class)
        if not self.has_residual:
            self.residual_proj = nn.Linear(in_ch, n_class)

    def reset_parameters(self):
        self.hgc1.reset_parameters()
        self.hgc2.reset_parameters()
        self.bn1.reset_parameters()
        if not self.has_residual:
            torch.nn.init.xavier_uniform_(self.residual_proj.weight)
            if self.residual_proj.bias is not None:
                torch.nn.init.zeros_(self.residual_proj.bias)

    def forward(self, data, return_emb=False):
        x = data.x
        G = data.edge_index

        # 确保G是Float类型
        if not G.is_floating_point():
            G = G.float()
        
        # 第一层带激活函数和批量归一化
        identity = x  # 保存输入用于残差连接
        
        x = self.hgc1(x, G)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 第二层
        outs = self.hgc2(x, G)
        
        # 残差连接
        if self.has_residual:
            outs = outs + identity
        else:
            outs = outs + self.residual_proj(identity)

        node_emb = x if return_emb else None
        return outs, node_emb, None


class HNHN(nn.Module):
    """
    """

    def __init__(self, args):
        super(HNHN, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout
        
        self.convs = nn.ModuleList()
        # two cases
        if self.num_layers == 1:
            self.convs.append(HNHNConv(args.num_features, args.MLP_hidden, args.num_classes,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
        else:
            self.convs.append(HNHNConv(args.num_features, args.MLP_hidden, args.MLP_hidden,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
            for _ in range(self.num_layers - 2):
                self.convs.append(HNHNConv(args.MLP_hidden, args.MLP_hidden, args.MLP_hidden,
                                           nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
            self.convs.append(HNHNConv(args.MLP_hidden, args.MLP_hidden, args.num_classes,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, return_emb=False):

        x = data.x
        
        if self.num_layers == 1:
            conv = self.convs[0]
            outs = conv(x, data)
            # x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = F.relu(conv(x, data))
                x = F.dropout(x, p=self.dropout, training=self.training)
            outs = self.convs[-1](x, data)

        node_emb = x if return_emb else None
        return outs, node_emb, None


class HCHA(nn.Module):
    """
    This model is proposed by "Hypergraph Convolution and Hypergraph Attention" (in short HCHA) and its convolutional layer 
    is implemented in pyg.
    """

    def __init__(self, args):
        super(HCHA, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.symdegnorm = args.HCHA_symdegnorm

#         Note that add dropout to attention is default in the original paper
        self.convs = nn.ModuleList()
        self.convs.append(HypergraphConv(args.num_features,
                                         args.MLP_hidden, self.symdegnorm))
        for _ in range(self.num_layers-2):
            self.convs.append(HypergraphConv(
                args.MLP_hidden, args.MLP_hidden, self.symdegnorm))
        # Output heads is set to 1 as default
        self.convs.append(HypergraphConv(
            args.MLP_hidden, args.num_classes, self.symdegnorm))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, return_emb=False):

        x = data.x
        edge_index = data.edge_index

        # 确保超边索引是连续的
        unique_edges = edge_index[1].unique()
        if unique_edges.size(0) != unique_edges.max() + 1:
            # 如果超边索引不连续，进行重新映射
            edge_mapping = {old.item(): new for new, old in enumerate(unique_edges)}
            new_edge_index = edge_index.clone()
            for i in range(edge_index.size(1)):
                new_edge_index[1, i] = edge_mapping[edge_index[1, i].item()]
            edge_index = new_edge_index

        for i, conv in enumerate(self.convs[:-1]):
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

#         x = F.dropout(x, p=self.dropout, training=self.training)
        outs = self.convs[-1](x, edge_index)
        node_emb = x if return_emb else None
        return outs, node_emb, None


class SetGNN(nn.Module):
    def __init__(self, args, norm=None):
        super(SetGNN, self).__init__()
        """
        args should contain the following:
        V_in_dim, V_enc_hid_dim, V_dec_hid_dim, V_out_dim, V_enc_num_layers, V_dec_num_layers
        E_in_dim, E_enc_hid_dim, E_dec_hid_dim, E_out_dim, E_enc_num_layers, E_dec_num_layers
        All_num_layers,dropout
        !!! V_in_dim should be the dimension of node features
        !!! E_out_dim should be the number of classes (for classification)
        """
#         V_in_dim = V_dict['in_dim']
#         V_enc_hid_dim = V_dict['enc_hid_dim']
#         V_dec_hid_dim = V_dict['dec_hid_dim']
#         V_out_dim = V_dict['out_dim']
#         V_enc_num_layers = V_dict['enc_num_layers']
#         V_dec_num_layers = V_dict['dec_num_layers']

#         E_in_dim = E_dict['in_dim']
#         E_enc_hid_dim = E_dict['enc_hid_dim']
#         E_dec_hid_dim = E_dict['dec_hid_dim']
#         E_out_dim = E_dict['out_dim']
#         E_enc_num_layers = E_dict['enc_num_layers']
#         E_dec_num_layers = E_dict['dec_num_layers']

#         Now set all dropout the same, but can be different
        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.InputNorm = args.deepset_input_norm
        self.GPR = args.GPR
        self.LearnMask = args.LearnMask
#         Now define V2EConvs[i], V2EConvs[i] for ith layers
#         Currently we assume there's no hyperedge features, which means V_out_dim = E_in_dim
#         If there's hyperedge features, concat with Vpart decoder output features [V_feat||E_feat]
        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()

        if self.LearnMask:
            self.Importance = Parameter(torch.ones(norm.size()))

        if self.All_num_layers == 0:
            self.classifier = MLP(in_channels=args.num_features,
                                  hidden_channels=args.Classifier_hidden,
                                  out_channels=args.num_classes,
                                  num_layers=args.Classifier_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)
        else:
            self.V2EConvs.append(HalfNLHconv(in_dim=args.num_features,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA,
                                             aggr=args.aggregate))
            self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA,
                                             aggr=args.aggregate))
            self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            for _ in range(self.All_num_layers-1):
                self.V2EConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA,
                                                 aggr=args.aggregate))
                self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
                self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA,
                                                 aggr=args.aggregate))
                self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            if self.GPR:
                self.MLP = MLP(in_channels=args.num_features,
                               hidden_channels=args.MLP_hidden,
                               out_channels=args.MLP_hidden,
                               num_layers=args.MLP_num_layers,
                               dropout=self.dropout,
                               Normalization=self.NormLayer,
                               InputNorm=False)
                self.GPRweights = Linear(self.All_num_layers+1, 1, bias=False)
                self.classifier = MLP(in_channels=args.MLP_hidden,
                                      hidden_channels=args.Classifier_hidden,
                                      out_channels=args.num_classes,
                                      num_layers=args.Classifier_num_layers,
                                      dropout=self.dropout,
                                      Normalization=self.NormLayer,
                                      InputNorm=False)
            else:
                self.classifier = MLP(in_channels=args.MLP_hidden,
                                      hidden_channels=args.Classifier_hidden,
                                      out_channels=args.num_classes,
                                      num_layers=args.Classifier_num_layers,
                                      dropout=self.dropout,
                                      Normalization=self.NormLayer,
                                      InputNorm=False)


#         Now we simply use V_enc_hid=V_dec_hid=E_enc_hid=E_dec_hid
#         However, in general this can be arbitrary.


    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()
        if self.GPR:
            self.MLP.reset_parameters()
            self.GPRweights.reset_parameters()
        if self.LearnMask:
            nn.init.ones_(self.Importance)

    def forward(self, data, return_emb=False):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
#             The data should contain the follows
#             data.x: node features
#             data.V2Eedge_index:  edge list (of size (2,|E|)) where
#             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges
#      
        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnMask:
            norm = self.Importance*norm
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0)
        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
#                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
#                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            outs = self.classifier(x)
        else:
            x = F.dropout(x, p=0.2, training=self.training) # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
#                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.E2VConvs[i](
                    x, reversed_edge_index, norm, self.aggr))
#                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            outs = self.classifier(x)

        node_emb = x if return_emb else None
        return outs, node_emb, None


class MLP_model(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, args, InputNorm=False):
        super(MLP_model, self).__init__()
        in_channels = args.num_features
        hidden_channels = args.MLP_hidden
        out_channels = args.num_classes
        num_layers = args.All_num_layers
        dropout = args.dropout
        Normalization = args.normalization

        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if normalization.__class__.__name__ != 'Identity':
                normalization.reset_parameters()

    def forward(self, data, return_emb=False):
        x = data.x
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        outs = self.lins[-1](x)
        node_emb = x if return_emb else None
        return outs, node_emb, None


"""
The code below is directly adapt from the official implementation of UniGNN.
"""
# NOTE: can not tell which implementation is better statistically 

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X



# v1: X -> XW -> AXW -> norm
class UniSAGEConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        
        # X0 = X # NOTE: reserved for skip connection

        X = self.W(X)

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.aggregate) # [E, C]

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce=self.args.second_aggregate, dim_size=N) # [N, C]
        X = X + Xv 

        if self.args.use_norm:
            X = normalize_l2(X)

        # NOTE: concat heads or mean heads?
        # NOTE: normalize here?
        # NOTE: skip concat here?

        return X



# v1: X -> XW -> AXW -> norm
class UniGINConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.eps = nn.Parameter(torch.Tensor([0.]))
        self.args = args 
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


    def forward(self, X, vertex, edges):
        N = X.shape[0]
        # X0 = X # NOTE: reserved for skip connection
        
        # v1: X -> XW -> AXW -> norm
        X = self.W(X) 

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.aggregate) # [E, C]
        
        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        X = (1 + self.eps) * X + Xv 

        if self.args.use_norm:
            X = normalize_l2(X)


        
        # NOTE: concat heads or mean heads?
        # NOTE: normalize here?
        # NOTE: skip concat here?

        return X



# v1: X -> XW -> AXW -> norm
class UniGCNConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args 
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        degE = self.args.degE
        degV = self.args.degV
        
        # v1: X -> XW -> AXW -> norm
        
        X = self.W(X)

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.aggregate) # [E, C]
        
        Xe = Xe * degE 

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        
        Xv = Xv * degV

        X = Xv 
        
        if self.args.use_norm:
            X = normalize_l2(X)

        # NOTE: skip concat here?

        return X



# v2: X -> AX -> norm -> AXW 
class UniGCNConv2(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=True)        
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args 
        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, X, vertex, edges):
        N = X.shape[0]
        degE = self.args.degE
        degV = self.args.degV

        # v3: X -> AX -> norm -> AXW 

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.aggregate) # [E, C]
        
        Xe = Xe * degE 

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        
        Xv = Xv * degV

        X = Xv 

        X = normalize_l2(X)


        X = self.W(X)


        # NOTE: result might be slighly unstable
        # NOTE: skip concat here?

        return X



class UniGATConv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2, skip_sum=True):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.att_v = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_drop  = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.skip_sum = skip_sum
        self.args = args
        self.reset_parameters()

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def reset_parameters(self):
        glorot(self.att_v)
        glorot(self.att_e)

    def forward(self, X, vertex, edges, return_edge_emb=False):
        H, C, N = self.heads, self.out_channels, X.shape[0]
        
        # X0 = X # NOTE: reserved for skip connection

        X0 = self.W(X)
        X = X0.view(N, H, C)

        Xve = X[vertex] # [nnz, H, C]
        # 把属于每个超边的节点嵌入按指定聚合方式聚合，得到该超边的语义表示
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.aggregate) # [E, H, C]


        alpha_e = (Xe * self.att_e).sum(-1) # [E, H, 1]
        a_ev = alpha_e[edges]
        alpha = a_ev # Recommed to use this
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, vertex, num_nodes=N)
        alpha = self.attn_drop( alpha )
        alpha = alpha.unsqueeze(-1)


        Xev = Xe[edges] # [nnz, H, C]
        Xev = Xev * alpha 
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, H, C]
        X = Xv 
        X = X.view(N, H * C)

        X = normalize_l2(X)

        if self.skip_sum:
            X = X + X0 

        # NOTE: concat heads or mean heads?
        # NOTE: skip concat here?

        return (X, Xe) if return_edge_emb else X




__all_convs__ = {
    'UniGAT': UniGATConv,
    'UniGCN': UniGCNConv,
    'UniGCN2': UniGCNConv2,
    'UniGIN': UniGINConv,
    'UniSAGE': UniSAGEConv,
}



class UniGNN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, nhead, V, E):
        """UniGNN

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        Conv = __all_convs__[args.model_name]
        self.conv_out = Conv(args, nhid * nhead, nclass, heads=1, dropout=args.attn_drop)
        self.convs = nn.ModuleList(
            [ Conv(args, nfeat, nhid, heads=nhead, dropout=args.attn_drop)] +
            [Conv(args, nhid * nhead, nhid, heads=nhead, dropout=args.attn_drop) for _ in range(nlayer-2)]
        )
        self.model_name = args.model_name
        self.V = V 
        self.E = E 
        # act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = nn.ReLU()
        self.input_drop = nn.Dropout(p=0.1)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, data, return_emb=False):
        V, E = data.edge_index[0], data.edge_index[1]
        # V, E = self.V, self.E 
        
        X = data.x

        X = self.input_drop(X)
        for conv in self.convs:
            if return_emb and self.model_name == 'UniGAT':
                # Xe 聚合关联的节点嵌入
                X, Xe = conv(X, V, E, return_edge_emb=True)
            else:
                X = conv(X, V, E)
            X = self.act(X)
            X = self.dropout(X)

        outs = self.conv_out(X, V, E)
        node_emb = X if return_emb else None
        edge_emb = Xe if (return_emb and self.model_name == 'UniGAT') else None
        return outs, node_emb, edge_emb


class UniGCNIIConv(nn.Module):
    def __init__(self, args, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.args = args

    def reset_parameters(self):
        self.W.reset_parameters()
        
    def forward(self, X, vertex, edges, alpha, beta, X0):
        N = X.shape[0]
        degE = self.args.UniGNN_degE
        degV = self.args.UniGNN_degV

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce='mean') # [E, C], reduce is 'mean' here as default
        
        Xe = Xe * degE 

        Xev = Xe[edges] # [nnz, C]
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N) # [N, C]
        
        Xv = Xv * degV
        
        X = Xv
        
        X = normalize_l2(X)

        Xi = (1-alpha) * X + alpha * X0
        X = (1-beta) * Xi + beta * self.W(Xi)


        return X



class UniGCNII(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, nhead, V, E):
        """UniGNNII

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        self.V = V 
        self.E = E 
        nhid = nhid * nhead
        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act['relu'] # Default relu
        self.input_drop = nn.Dropout(0.6) # 0.6 is chosen as default
        self.dropout = nn.Dropout(0.2) # 0.2 is chosen for GCNII

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(nfeat, nhid))
        for _ in range(nlayer):
            self.convs.append(UniGCNIIConv(args, nhid, nhid))
        self.convs.append(torch.nn.Linear(nhid, nclass))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        self.dropout = nn.Dropout(0.2) # 0.2 is chosen for GCNII
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
    def forward(self, data, return_emb=False):
        x = data.x
        V, E = self.V, self.E 
        lamda, alpha = 0.5, 0.1 
        x = self.dropout(x)
        x = F.relu(self.convs[0](x))
        x0 = x 
        for i,con in enumerate(self.convs[1:-1]):
            x = self.dropout(x)
            beta = math.log(lamda/(i+1)+1)
            x = F.relu(con(x, V, E, alpha, beta, x0))
        x = self.dropout(x)
        outs = self.convs[-1](x)
        node_emb = x if return_emb else None
        return outs, node_emb, None
