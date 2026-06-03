#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

"""
This script contains functions for loading the following datasets:
        co-authorship: (dblp, cora)
        walmart-trips (From cornell)
        Amazon-reviews
        U.S. House committee
"""

import torch
import os
import pickle

import os.path as osp
import numpy as np
import scipy.sparse as sp

from torch_geometric.data import Data
from torch_sparse import coalesce
# from randomperm_code import random_planetoid_splits
from sklearn.feature_extraction.text import CountVectorizer


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required only for Yelp/Cornell-style dataset loaders.") from exc
    return pd


def load_LE_dataset(path=None, dataset="ModelNet40", train_percent = 0.025):
    # load edges, features, and labels.
    print('Loading {} dataset...'.format(dataset))
    
    file_name = f'{dataset}.content'
    p2idx_features_labels = osp.join(path, dataset, file_name)
    # 使用 numpy 的 genfromtxt 函数读取特征和标签文件，数据类型为字符串
    idx_features_labels = np.genfromtxt(p2idx_features_labels,
                                        dtype=np.dtype(str))
    # features = np.array(idx_features_labels[:, 1:-1])
    # 将特征数据转换为稀疏矩阵格式，数据类型为 32 位浮点数
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    # 将标签数据转换为 LongTensor 格式
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))

    print ('load features')

    # build graph
    # 提取节点 ID
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 创建一个字典，将每个节点 ID 映射到其在数组中的索引
    idx_map = {j: i for i, j in enumerate(idx)}
    
    file_name = f'{dataset}.edges'
    p2edges_unordered = osp.join(path, dataset, file_name)
    # 使用 numpy 的 genfromtxt 函数读取边文件，数据类型为 32 位整数
    edges_unordered = np.genfromtxt(p2edges_unordered,
                                    dtype=np.int32)
    
    # 将边数据中的节点 ID 映射到其在数组中的索引
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    print ('load edges')

    # 将特征数据转换为 FloatTensor 格式
    projected_features = torch.FloatTensor(np.array(features.todense()))

    
    # From adjacency matrix to edge_list
    # 从邻接矩阵转换为边列表
    edge_index = edges.T 
#     ipdb.set_trace()
    # 断言边索引的最大值等于边索引的最小值减一
    assert edge_index[0].max() == edge_index[1].min() - 1

    # check if values in edge_index is consecutive. i.e. no missing value for node_id/he_id.
    # 检查边索引中的值是否连续，即没有缺失的节点 ID 或超边 ID
    assert len(np.unique(edge_index)) == edge_index.max() + 1
    
    # 计算节点数量
    num_nodes = edge_index[0].max() + 1
    # 计算超边数量
    num_he = edge_index[1].max() - num_nodes + 1
    
    # 将边索引堆叠，包括其转置，以确保图是无向的
    edge_index = np.hstack((edge_index, edge_index[::-1, :]))
    # ipdb.set_trace()
    
    # build torch data class
    # 构建 PyTorch Geometric 的 Data 类对象
    data = Data(
#             x = projected_features, 
            x = torch.FloatTensor(np.array(features[:num_nodes].todense())), 
            edge_index = torch.LongTensor(edge_index),
            y = labels[:num_nodes])

    
    # ipdb.set_trace()
    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    # 使用 coalesce 函数处理边索引，去除重复边并排序
    total_num_node_id_he_id = len(np.unique(edge_index))
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
            



#     ipdb.set_trace()
    
#     # generate train, test, val mask.
    # 计算节点数量
    n_x = num_nodes
#     n_x = n_expanded
    # 计算类别数量
    num_class = len(np.unique(labels[:num_nodes].numpy()))
    # 计算每个类别的训练样本数量
    val_lb = int(n_x * train_percent)
    # 计算每个类别的训练样本数量（四舍五入）
    percls_trn = int(round(train_percent * n_x / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.n_x = n_x
    # add parameters to attribute
    
    
    data.train_percent = train_percent
    data.num_hyperedges = num_he
    
    return data

def load_citation_dataset(path='../hyperGCN/data/', dataset = 'cora', train_percent = 0.025):
    '''
    this will read the citation dataset from HyperGCN, and convert it edge_list to 
    [[ -V- | -E- ]
     [ -E- | -V- ]]
    '''
    print(f'Loading hypergraph dataset from hyperGCN: {dataset}')

    # first load node features:
    with open(osp.join(path, dataset, 'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        features = features.todense()

    # then load node labels:
    with open(osp.join(path, dataset, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    with open(osp.join(path, dataset, 'hypergraph.pickle'), 'rb') as f:
        # hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f'number of hyperedges: {len(hypergraph)}')

    edge_idx = num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    edge_index = np.array([ node_list + edge_list,
                            edge_list + node_list], dtype = np.int32)
    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
            

    n_x = num_nodes
#     n_x = n_expanded
    num_class = len(np.unique(labels.numpy()))
    val_lb = int(n_x * train_percent)
    percls_trn = int(round(train_percent * n_x / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.n_x = n_x
    # add parameters to attribute
    
    data.train_percent = train_percent
    data.num_hyperedges = len(hypergraph)
    
    return data

def load_yelp_dataset(path='./data/raw_data/yelp_raw_datasets/', dataset = 'yelp', 
        name_dictionary_size = 1000,
        train_percent = 0.025):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to 
    [[ -V- | -E- ]
     [ -E- | -V- ]]

    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.

    node features:
        - latitude, longitude
        - state, in one-hot coding. 
        - city, in one-hot coding. 
        - name, in bag-of-words

    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    '''
    print(f'Loading hypergraph dataset from {dataset}')
    pd = _require_pandas()

    # first load node features:
    # load longtitude and latitude of restaurant.
    latlong = pd.read_csv(osp.join(path, 'yelp_restaurant_latlong.csv')).values

    # city - zipcode - state integer indicator dataframe.
    loc = pd.read_csv(osp.join(path, 'yelp_restaurant_locations.csv'))
    state_int = loc.state_int.values
    city_int = loc.city_int.values

    num_nodes = loc.shape[0]
    state_1hot = np.zeros((num_nodes, state_int.max()))
    state_1hot[np.arange(num_nodes), state_int - 1] = 1

    city_1hot = np.zeros((num_nodes, city_int.max()))
    city_1hot[np.arange(num_nodes), city_int - 1] = 1

    # convert restaurant name into bag-of-words feature.
    vectorizer = CountVectorizer(max_features = name_dictionary_size, stop_words = 'english', strip_accents = 'ascii')
    res_name = pd.read_csv(osp.join(path, 'yelp_restaurant_name.csv')).values.flatten()
    name_bow = vectorizer.fit_transform(res_name).todense()

    features = np.hstack([latlong, state_1hot, city_1hot, name_bow])

    # then load node labels:
    df_labels = pd.read_csv(osp.join(path, 'yelp_restaurant_business_stars.csv'))
    labels = df_labels.values.flatten()

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    # Yelp restaurant review hypergraph is store in a incidence matrix.
    H = pd.read_csv(osp.join(path, 'yelp_restaurant_incidence_H.csv'))
    node_list = H.node.values - 1
    edge_list = H.he.values - 1 + num_nodes

    edge_index = np.vstack([node_list, edge_list])
    edge_index = np.hstack([edge_index, edge_index[::-1, :]])

    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
            

    n_x = num_nodes
#     n_x = n_expanded
    num_class = len(np.unique(labels.numpy()))
    val_lb = int(n_x * train_percent)
    percls_trn = int(round(train_percent * n_x / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.n_x = n_x
    # add parameters to attribute
    
    data.train_percent = train_percent
    data.num_hyperedges = H.he.values.max()
    
    return data

def load_tencent_2k_dataset(path='./data/AllSet_all_raw_data/', dataset='tencent_2k', train_percent=0.025):
    """
    加载 tencent_2k 数据集并构建 PyG 的 Data 对象
    数据包括：
    - edge_list.pkl：每个 tuple 是一个超边，包含多个节点 ID
    - features.pkl：CSR 稀疏格式，shape = [num_nodes, num_features]
    - labels.pkl：shape = [num_nodes]
    """

    print(f"Loading tencent_2k dataset from {path}")
    
    # 加载 features
    with open(os.path.join(path, 'features.pkl'), 'rb') as f:
        features = pickle.load(f)
    
    # 加载 labels
    with open(os.path.join(path, 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)
    
    # 加载 edge_list
    with open(os.path.join(path, 'edge_list.pkl'), 'rb') as f:
        edge_list = pickle.load(f)
    
    # 如果 features 是稀疏矩阵，转换为其密集形式
    if hasattr(features, 'todense'):
        features = features.todense()
    # 转换为 PyTorch tensor
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    # 构建对称的 edge_index
    node_list = []
    edge_list_new = []
    edge_idx = features.shape[0]  # 假设超边的 ID 从节点数开始计数
    
    for he in edge_list:
        cur_he = he
        cur_size = len(cur_he)
        
        node_list += list(cur_he)
        edge_list_new += [edge_idx] * cur_size
        
        edge_idx += 1
    
    # 对称的 edge_index
    edge_index = np.array([node_list + edge_list_new, edge_list_new + node_list], dtype=np.int32)
    edge_index = torch.LongTensor(edge_index)
    
    num_nodes = features.shape[0]
    num_classes = len(torch.unique(labels))  # 类别数
    
    # 创建 Data 对象
    data = Data(x=features, edge_index=edge_index, y=labels)
    
    # 添加一些额外的属性（例如：num_hyperedges 和 n_x）
    data.num_hyperedges = len(edge_list)  # 超边的数量
    data.n_x = torch.tensor([num_nodes])
    
    n_x = num_nodes
#     n_x = n_expanded
    num_class = len(np.unique(labels.numpy()))
    val_lb = int(n_x * train_percent)
    percls_trn = int(round(train_percent * n_x / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.n_x = n_x
    # add parameters to attribute
    
    data.train_percent = train_percent
    return data

def load_cornell_dataset(path='./data/raw_data/', dataset = 'amazon', 
        feature_noise = 0.1,
        feature_dim = None,
        train_percent = 0.025):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to 
    [[ -V- | -E- ]
     [ -E- | -V- ]]

    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.

    node features:
        - add gaussian noise with sigma = nosie, mean = one hot coded label.

    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    '''
    print(f'Loading hypergraph dataset from cornell: {dataset}')
    pd = _require_pandas()

    # first load node labels
    df_labels = pd.read_csv(osp.join(path, dataset, f'node-labels-{dataset}.txt'), names = ['node_label'])
    num_nodes = df_labels.shape[0]
    labels = df_labels.values.flatten()

    # then create node features.
    num_classes = df_labels.values.max()
    features = np.zeros((num_nodes, num_classes))

    features[np.arange(num_nodes), labels - 1] = 1
    if feature_dim is not None:
        num_row, num_col = features.shape
        zero_col = np.zeros((num_row, feature_dim - num_col), dtype = features.dtype)
        features = np.hstack((features, zero_col))

    features = np.random.normal(features, feature_noise, features.shape)
    print(f'number of nodes:{num_nodes}, feature dimension: {features.shape[1]}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    # Corenll datasets are stored in lines of hyperedges. Each line is the set of nodes for that edge.
    p2hyperedge_list = osp.join(path, dataset, f'hyperedges-{dataset}.txt')
    node_list = []
    he_list = []
    he_id = num_nodes

    with open(p2hyperedge_list, 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            cur_set = line.split(',')
            cur_set = [int(x) for x in cur_set]

            node_list += cur_set
            he_list += [he_id] * len(cur_set)
            he_id += 1
    # shift node_idx to start with 0.
    node_idx_min = np.min(node_list)
    node_list = [x - node_idx_min for x in node_list]

    edge_index = [node_list + he_list, 
                  he_list + node_list]

    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
            

    n_x = num_nodes
#     n_x = n_expanded
    num_class = len(np.unique(labels.numpy()))
    val_lb = int(n_x * train_percent)
    percls_trn = int(round(train_percent * n_x / num_class))
    # data = random_planetoid_splits(data, num_class, percls_trn, val_lb)
    data.n_x = n_x
    # add parameters to attribute
    
    data.train_percent = train_percent
    data.num_hyperedges = he_id - num_nodes
    
    return data

if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()
    # data = load_yelp_dataset()
    data = load_cornell_dataset(dataset = 'walmart-trips', feature_noise = 0.1)
    data = load_cornell_dataset(dataset = 'walmart-trips', feature_noise = 1)
    data = load_cornell_dataset(dataset = 'walmart-trips', feature_noise = 10)

