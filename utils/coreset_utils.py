# Codes adapted from https://github.com/Amanda-Zheng/SFGC

import torch
import numpy as np
from collections import Counter
import logging


class Base:
    def __init__(self, data, train_idx, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device
        self.train_idx = train_idx
        # 检查训练样本数量是否足够，如果不够，则减少比例
        if hasattr(data, 'train_mask'):
            train_mask_ratio = data.train_mask.sum() / data.n_x
        else:
            train_mask_ratio = len(train_idx) / data.n_x

        assert len(train_idx) > data.n_x * args.reduction_rate, f'Too few training samples, reduction rate should be smaller than {train_mask_ratio:.4f}'

        reduce_numbers = int(data.n_x * args.reduction_rate)
        # 根据训练集中的标签分布计算每个类别应该保留的样本数
        self.num_class_dict = self.condensed_node_class2num(data.y[train_idx].cpu().tolist(), reduce_numbers)
        logging.info(f'reduce training number {len(train_idx)} to {reduce_numbers} samples')

    @staticmethod
    def condensed_node_class2num(labels, reduce_numbers):
        # 统计每个类别出现的次数，形成类别:数量的字典
        num_class_dict = Counter(labels)
        # 计算所有节点总数
        all_num = sum(num_class_dict.values())
        # 将每个类别的数量转化为占总节点数的比例
        for c, num in num_class_dict.items():
            num_class_dict[c] = num / all_num

        condense_class_dict = {}
        # 遍历每个类别和其比例
        for i, (c, ratio) in enumerate(num_class_dict.items()):
            # 除了最后一个类别，都按比例分配（至少为1）
            if i != len(num_class_dict) - 1:
                condense_class_dict[c] = max(int(ratio * reduce_numbers), 1)
            # 最后一个类别分配剩余所有数量，保证总数精确等于reduce_numbers
            else:
                condense_class_dict[c] = reduce_numbers - sum(condense_class_dict.values())
        return condense_class_dict


class KCenter(Base):
    """
    基于 K-Center 算法的数据选择方法，主要思想是每次选择离当前已选择中心点最远的样本作为新的中心点。
    """
    def __init__(self, data, train_idx, args, device='cuda', **kwargs):
        super(KCenter, self).__init__(data, train_idx, args, device='cuda', **kwargs)

    def select(self, embeds, inductive=False):
        # feature: embeds
        # kcenter # class by class
        num_class_dict = self.num_class_dict
        train_labels = self.data.y[self.train_idx]
        # labels_train = self.data.labels_train
        idx_selected = []
        for class_id, cnt in num_class_dict.items():
            idx = self.train_idx[train_labels == class_id]
            feature = embeds[idx]
            mean = torch.mean(feature, dim=0, keepdim=True)
            # dis = distance(feature, mean)[:,0]
            dis = torch.cdist(feature, mean)[:, 0]
            rank = torch.argsort(dis)
            idx_centers = rank[:1].tolist()
            for _ in range(int(cnt)-1):
                feature_centers = feature[idx_centers]
                dis_center = torch.cdist(feature, feature_centers)
                dis_min, _ = torch.min(dis_center, dim=-1)
                id_max = torch.argmax(dis_min).item()
                idx_centers.append(id_max)
            idx_selected.append(idx[idx_centers])
        # return np.array(idx_selected).reshape(-1)
        return torch.cat(idx_selected)


class Herding(Base):
    """
    基于 Herding 算法的数据选择方法，核心思想是逐步选择样本，每次选择距离当前已选择样本均值最小的样本。
    """
    def __init__(self, data, train_idx, args, device='cuda', **kwargs):
        super(Herding, self).__init__(data, train_idx, args, device='cuda', **kwargs)

    def select(self, embeds, inductive=False):
        num_class_dict = self.num_class_dict
        train_labels = self.data.y[self.train_idx]

        idx_selected = []

        # herding # class by class
        for class_id, cnt in num_class_dict.items():
            idx = self.train_idx[train_labels == class_id]
            features = embeds[idx]
            mean = torch.mean(features, dim=0, keepdim=True)
            selected = []
            idx_left = np.arange(features.shape[0]).tolist()

            for i in range(cnt):
                try:
                    det = mean*(i+1) - torch.sum(features[selected], dim=0)
                    dis = torch.cdist(det, features[idx_left])
                    id_min = torch.argmin(dis)
                    selected.append(idx_left[id_min])
                    del idx_left[id_min]
                except:
                    continue
            idx_selected.append(idx[selected])
        return torch.cat(idx_selected)


class Random(Base):
    """
    Random 类实现了一个简单的随机选择策略，每个类别随机选择样本。
    """
    def __init__(self, data, train_idx, args, device='cuda', **kwargs):
        super(Random, self).__init__(data, train_idx, args, device='cuda', **kwargs)

    def select(self, embeds, inductive=False):
        num_class_dict = self.num_class_dict
        
        train_labels = self.data.y[self.train_idx]
        idx_selected = []

        for class_id, cnt in num_class_dict.items():
            idx = self.train_idx[train_labels == class_id].cpu().numpy()
            selected = np.random.permutation(idx)
            idx_selected.append(torch.from_numpy(selected[:cnt]))

        return torch.cat(idx_selected)


