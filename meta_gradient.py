import logging
import os
import random
import copy
import scipy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import deeprobust.graph.utils as utils
from copy import deepcopy
from tqdm import tqdm
from utils.dataloader import load_data
from model.model_loader import parse_method
from model.reparam_module import ReparamModule
from torch_geometric.data import Data
from torch_geometric.utils import degree


class MetaGtt:
    """
    基于图神经切线核的图蒸馏类(转导式版本)
      
    属性:
        data: 数据集对象
        args: 参数对象
        device: 计算设备
        labels_syn: 合成标签
        feat_syn: 合成特征
        optimizer_feat: 特征优化器
    """

    def __init__(self, data, split_idx_lst, args, device="cuda", **kwargs):
        self.data = data
        self.args = args
        self.device = device
        self.split_idx_lst = split_idx_lst
        split_idx = random.choice(split_idx_lst)
        train_idx, val_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
        self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx

        features, edge_index, labels = data.x, data.edge_index, data.y
        # 根据特征、边、标签生成核心子集
        feat_init, edge_index_init, labels_init = self.get_coreset_init(features, edge_index, labels, args, run=args.runs)
        
        # 转 Tensor 并移动到设备
        feat_init = feat_init.to(device)
        edge_index_init = edge_index_init.to(device)
        labels_init = labels_init.to(device)

        # 保存合成特征和标签
        self.feat_syn = nn.Parameter(feat_init.clone())
        self.labels_syn = labels_init.clone()
        self.edge_index_syn = edge_index_init.clone()
        self.labels_init = labels_init.clone()

        print(type(self.edge_index_syn))
        print(self.edge_index_syn.shape)
        # 构造 PyG 格式的数据（用于后续软标签生成）
        if args.method == 'HGNN':
            # 对于HGNN模型，需要特殊处理为G矩阵
            num_nodes = feat_init.shape[0]
            # 创建单位矩阵作为G矩阵（用于HGNN模型）
            G_matrix = torch.eye(num_nodes).to(device)
            # 构造专用于HGNN的data对象
            data_4_soft = Data(
                x=self.feat_syn,
                edge_index=G_matrix,  # 对HGNN模型，这里使用G矩阵
                y=self.labels_syn
            )
        else:
            # 对于其他模型，使用标准的edge_index
            data_4_soft = Data(
                x=self.feat_syn,
                edge_index=self.edge_index_syn,
                y=self.labels_syn
            )
        # 不再添加is_distill标记
        self.data_4_soft = add_norm_to_data(data_4_soft)
        self.data_4_soft.num_nodes = self.feat_syn.shape[0]
        
        data_all = Data(
            x = features.clone().detach().to(torch.float),
            edge_index = edge_index.clone().detach().to(torch.long),
            y = labels.clone().detach().to(torch.long)
        ).to(self.device)
        self.data_all = add_norm_to_data(data_all)
        self.data_all.idx_test = test_idx

        # 初始化优化器
        if args.optimizer_con == "Adam":
            self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        elif args.optimizer_con == "SGD":
            self.optimizer_feat = torch.optim.SGD([self.feat_syn], lr=args.lr_feat, momentum=0.9)

        logging.info("feat_syn: {}".format(self.feat_syn.shape))

    def beta_mapping(self, epoch, upper_bound, lower_bound, end_epoch):
        """
        将训练轮数映射到特定范围
        
        使用sigmoid函数将训练轮数映射到[lower_bound, upper_bound]范围内。
        当epoch >= end_epoch时，返回lower_bound。
        
        Args:
            epoch: 当前训练轮数
            upper_bound: 映射范围的上界
            lower_bound: 映射范围的下界
            end_epoch: 结束轮数
            
        Returns:
            float: 映射后的值
        """
        if epoch >= end_epoch:
            return lower_bound

        x = epoch / end_epoch
        mapped_value = lower_bound + (upper_bound - lower_bound) / (
            1 + np.exp(-10 * (x - 0.5))
        )

        return mapped_value

    def expert_load(self):
        """
        加载专家模型轨迹
        
        从指定目录加载专家模型的训练轨迹，支持两种模式:
        1. 加载所有专家轨迹(load_all=True)
        2. 加载部分专家轨迹(load_all=False)
        
        Returns:
            tuple: (file_idx, expert_idx, expert_files)
                - file_idx: 当前文件索引
                - expert_idx: 当前专家索引
                - expert_files: 专家文件列表
        """
        args = self.args
        expert_dir = args.buffer_path
        logging.info("Expert Dir: {}".format(expert_dir))

        if args.load_all:
            # 加载所有专家轨迹
            buffer = []
            n = 0
            while os.path.exists(
                os.path.join(expert_dir, f"replay_buffer_{n}.pt")
            ):
                buffer = buffer + torch.load(
                    os.path.join(expert_dir, f"replay_buffer_{n}.pt")
                )
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(expert_dir))

        else:
            # 加载部分专家轨迹
            expert_files = []
            n = 0
            while os.path.exists(
                os.path.join(expert_dir, f"replay_buffer_{n}.pt")
            ):
                expert_files.append(
                    os.path.join(expert_dir, f"replay_buffer_{n}.pt")
                )
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(expert_dir))
            file_idx = 0
            expert_idx = 0
            random.shuffle(expert_files)
            if args.max_files is not None:
                expert_files = expert_files[: args.max_files]
            print("loading file {}".format(expert_files[file_idx]))
            buffer = torch.load(expert_files[file_idx])
            if args.max_experts is not None:
                buffer = buffer[: args.max_experts]
            random.shuffle(buffer)
            self.buffer = buffer

        return file_idx, expert_idx, expert_files

    def synset_save(self):
        """
        保存合成数据集 (超图版本)

        将当前训练好的合成数据集(特征、超图关联矩阵H、标签)保存下来。
        支持软标签和硬标签两种模式。
        """
        args = self.args

        with torch.no_grad():
            feat_save = self.feat_syn
            eval_labs = self.labels_syn

        feat_syn_eval = copy.deepcopy(feat_save.detach())
        label_syn_eval = copy.deepcopy(eval_labs.detach())

        # 超图情况下，简单用单位矩阵代替（可自定义设计超图结构）
        H_syn_eval = torch.eye(feat_syn_eval.shape[0]).to(self.device)

        return feat_syn_eval, H_syn_eval, label_syn_eval

    def eval_synset(self, args):
        """
        评估合成数据集
        
        使用合成数据集训练模型并评估其性能。支持多种评估指标和模型类型。
        
        Args:
            args: 参数对象，包含评估相关的配置
            
        Returns:
            tuple: (res_val, res_test)
                - res_val: 验证集结果
                - res_test: 测试集结果
        """
        # 设置随机种子以确保结果可复现
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        device = torch.device(args.device)
        # logging.info('start!')
        if args.dname in ["cora", "citeseer"]:
            args.epsilon = 0.05
        else:
            args.epsilon = 0.01

        data = self.data

        # 初始化结果列表
        res_val = []
        res_test = []
        nlayer = 2

        # 多次运行评估
        for i in range(args.nruns):
            best_acc_val, best_acc_test = self.test(
                args, data, device, model_type=args.method, nruns=i
            )

            res_val.append(best_acc_val)
            res_test.append(best_acc_test)

        # 计算统计结果
        res_val = np.array(res_val)
        res_test = np.array(res_test)
        logging.info("Model:{}, Layer: {}".format(args.method, nlayer))
        logging.info(
            "TEST: Full Graph Mean Accuracy: {:.6f}, STD: {:.6f}".format(
                res_test.mean(), res_test.std()
            )
        )
        logging.info(
            "TEST: Valid Graph Mean Accuracy: {:.6f}, STD: {:.6f}".format(
                res_val.mean(), res_val.std()
            )
        )

        return res_val, res_test

    def SoftCrossEntropy(self, inputs, target, reduction="average"):
        """
        计算SoftCrossEntropy损失
        
        用于软标签训练时的损失计算，将输入与目标软标签进行比较。
        
        Args:
            inputs: 模型输出的logits
            target: 目标软标签
            reduction: 损失计算方式，默认为"average"
            
        Returns:
            tensor: 计算得到的损失值
        """
        input_log_likelihood = -inputs
        target_log_likelihood = F.softmax(target, dim=1)
        batch = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
        return loss

    def test(self, args, data, device, model_type, nruns):
        """
        测试模型性能 (超图版本)

        Args:
            args: 参数对象
            data: 超图数据集对象
            device: 计算设备
            model_type: 模型类型
            nruns: 当前运行编号
        Returns:
            tuple: (best_acc_val, best_acc_test)
        """
        # 准备合成数据
        if args.whole_data != 1:
            feat_syn, _, labels_syn = self.synset_save()  # 不再需要 H_syn
            feat_syn, labels_syn = feat_syn.to(device), labels_syn.to(device) 
            # 构造自环 edge_index
            num_nodes = feat_syn.shape[0]
            edge_index_syn = torch.arange(num_nodes, device=device)
            edge_index_syn = torch.stack([edge_index_syn, edge_index_syn], dim=0)  # [2, num_nodes]

            # 构造 PyG Data 对象
            if args.method == 'HGNN':
                # 对于HGNN模型，使用单位矩阵作为G矩阵
                G_matrix = torch.eye(num_nodes).to(device)
                data_syn = Data(x=feat_syn.clone(), edge_index=G_matrix, y=labels_syn).to(device)
            else:
                # 对于其他模型，使用标准的edge_index
                data_syn = Data(x=feat_syn.clone(), edge_index=edge_index_syn.clone(), y=labels_syn).to(device)
            
            # 添加标记，表明这是评估中的简化图结构
            data_syn = add_norm_to_data(data_syn)

            weight_decay = args.test_wd
            lr = args.test_lr_model

        else:
            logging.info("THIS IS THE ORIGINAL WHOLE DATA...")
            pass

        # 设置dropout率
        dropout = args.test_dropout

        # 初始化模型
        model = parse_method(args, data_syn)
        model = model.to(self.device)

        # 选择优化器
        if args.test_opt_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif args.test_opt_type == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay)

        # 初始化记录最佳结果
        best_acc_val = best_acc_test = best_acc_it = 0
        train_iters = args.test_model_iters

        patience = args.early_patience
        wait = 0

        for i in range(train_iters):
            # 学习率衰减
            if i == train_iters // 2 and args.lr_decay == 1:
                lr = args.test_lr_model * 0.5
                if args.test_opt_type == "Adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                elif args.test_opt_type == "SGD":
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

            # 前向传播和损失计算
            model.train()
            optimizer.zero_grad()

            output_syn = model.forward(data_syn)
            if isinstance(output_syn, tuple):
                output_syn = output_syn[0]

            if args.whole_data == 1:
                # 对于硬标签，直接使用 CrossEntropyLoss（输入为 logits）
                loss_train = F.cross_entropy(output_syn[self.train_idx], labels_syn[self.train_idx])
                acc_syn = utils.accuracy(output_syn[self.train_idx], labels_syn[self.train_idx])
            else:
                if args.soft_label:
                    output_syn = F.log_softmax(output_syn, dim=1)
                    labels_syn_log = F.log_softmax(labels_syn, dim=1)
                    loss_train = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)(output_syn, labels_syn_log)
                    acc_syn = utils.accuracy(output_syn, torch.argmax(labels_syn, dim=1))
                else:
                    # 纠正：当 soft_label 关闭时采用 CrossEntropyLoss
                    loss_train = F.cross_entropy(output_syn, labels_syn)
                    acc_syn = utils.accuracy(output_syn, labels_syn)

            loss_train.backward()
            optimizer.step()

            # 定期评估
            if i % 20 == 0:
                model.eval()
                labels_test = data.y[self.test_idx].to(device)
                labels_val = data.y[self.val_idx].to(device)

                output = model(self.data_all)
                if isinstance(output, tuple):
                    output = output[0]

                acc_val = utils.accuracy(output[self.val_idx], labels_val)
                acc_test = utils.accuracy(output[self.test_idx], labels_test)

                if acc_val.item() > best_acc_val + 1e-6:      # 有显著提升
                    best_acc_val  = acc_val.item()
                    best_acc_test = acc_test.item()
                    best_acc_it   = i
                    wait = 0                                   # 归零
                else:                                          # 无提升
                    wait += 1
                    if wait >= patience:
                        logging.info(f'Early-stop in test() at epoch {i}; '
                                    f'best_val={best_acc_val:.4f}')
                        break

        logging.info("FINAL BEST ACC TEST: {:.6f} within {}-iteration".format(best_acc_test, best_acc_it))
        return best_acc_val, best_acc_test


    def distill(self, writer):
        """
        执行蒸馏过程 (超图版本)

        主要包括：
        1. 初始化合成数据
        2. 加载专家轨迹
        3. 软标签处理（可选）
        """
        args = self.args
        data = self.data
        labels_init = self.labels_init
 
        # 加载专家轨迹
        file_idx, expert_idx, expert_files = self.expert_load()

        # 软标签处理
        if args.soft_label:
            
            model_4_soft = parse_method(args, data)
            model_4_soft = ReparamModule(model_4_soft)
            model_4_soft = model_4_soft.to(self.device)

            # 生成软标签
            model_4_soft.eval()
            Temp_params = self.buffer[0][-1]
            Initialize_Labels_params = torch.cat([p.data.to(args.device).reshape(-1) for p in Temp_params], 0)

            data_4_soft = self.data_4_soft
            label_soft, _, _ = model_4_soft.forward(data_4_soft, flat_param=Initialize_Labels_params)

            # 调整软标签
            labels_init = self.labels_init
            max_pred, pred_lab = torch.max(label_soft, dim=1)
            label_soft_new = label_soft.clone()
            for i in range(labels_init.shape[0]):
                if pred_lab[i] != labels_init[i]:
                    label_soft_new[i][labels_init[i]] = max_pred[i]
            # 后续用 label_soft_new 替换 label_soft
            self.labels_syn = (copy.deepcopy(label_soft_new.detach()).to(args.device).requires_grad_(True))

            self.labels_syn.requires_grad = True

            # 计算初始准确率
            acc = np.sum(
                np.equal(
                    np.argmax(label_soft.cpu().data.numpy(), axis=-1),
                    labels_init.cpu().data.numpy(),
                )
            )
            print("InitialAcc: {}".format(acc / len(self.labels_syn)))

            # 初始化软标签优化器
            self.optimizer_label = torch.optim.SGD(
                [self.labels_syn], lr=args.lr_y, momentum=0.9
            )

        else:
            self.labels_syn = labels_init

        self.syn_lr = torch.tensor(args.lr_student).to(self.device)

        if args.optim_lr == 1:
            self.syn_lr = self.syn_lr.detach().to(self.device).requires_grad_(True)
            if args.optimizer_lr == "Adam":
                optimizer_lr = torch.optim.Adam([self.syn_lr], lr=args.lr_lr)
            elif args.optimizer_lr == "SGD":
                optimizer_lr = torch.optim.SGD(
                    [self.syn_lr], lr=args.lr_lr, momentum=0.5
                )

        # 设置评估间隔和模型池
        eval_it_pool = np.arange(0, args.ITER + 1, args.eval_interval).tolist()
        from utils import get_eval_pool
        model_eval_pool = get_eval_pool(args.eval_type, args.method, args.method)
        accs_all_exps = dict()  # record performances of all experiments
        for key in model_eval_pool:
                accs_all_exps[key] = []

        best_accs_test = {m: 0 for m in model_eval_pool}
        best_accs_test_iter = {m: 0 for m in model_eval_pool}
        best_model_std_test = {m: 0 for m in model_eval_pool}

        best_loss = 1.0
        best_loss_it = 0

        patience_outer = args.outer_patience     # 新 argparse 参数
        wait_outer     = 0                       # 外层等待计数器

        # 主训练循环
        for it in range(0, args.ITER + 1):
            model = parse_method(args, data)
            model_4_clom = parse_method(args, data)
            
            model = ReparamModule(model)
            model_4_clom = ReparamModule(model_4_clom)

            model = model.to(self.device)
            model_4_clom= model_4_clom.to(self.device)

            model.train()

            # 计算总参数量
            num_params = sum([np.prod(p.size()) for p in model.parameters()])

            # 获取专家轨迹
            if args.load_all:
                expert_trajectory = self.buffer[np.random.randint(0, len(self.buffer))]
            else:
                expert_trajectory = self.buffer[expert_idx]
                expert_idx += 1
                if expert_idx == len(self.buffer):
                    expert_idx = 0
                    file_idx += 1
                    if file_idx == len(expert_files):
                        file_idx = 0
                        random.shuffle(expert_files)
                    print("loading file {}".format(expert_files[file_idx]))
                    if args.max_files != 1:
                        del self.buffer
                        self.buffer = torch.load(expert_files[file_idx])
                    if args.max_experts is not None:
                        self.buffer = self.buffer[:args.max_experts]
                    random.shuffle(self.buffer)

            # 设置起始epoch范围
            if args.expanding_window:
                Upper_Bound = args.max_start_epoch_s + it
                Upper_Bound = min(Upper_Bound, args.max_start_epoch)
            else:
                Upper_Bound = args.max_start_epoch

            print(Upper_Bound)

            np.random.seed(it)
            print(f"min_start_epoch: {args.min_start_epoch}, Upper_Bound: {Upper_Bound}")
            start_epoch = np.random.randint(args.min_start_epoch, Upper_Bound)

            np.random.seed(args.seed)

            # 轨迹保存间隔为 10 epoch，因此索引需 // 10
            start_epoch = start_epoch // 10
            starting_params = expert_trajectory[start_epoch]

            # 获取target参数
            if args.interval_buffer == 1:
                target_index = start_epoch + args.expert_epochs // 10
                if target_index >= len(expert_trajectory):
                    print(f"Warning: target index {target_index} exceeds trajectory length {len(expert_trajectory)}. Clamping to last index.")
                    target_index = len(expert_trajectory) - 1
                target_params = expert_trajectory[target_index]
                print(start_epoch + args.expert_epochs // 10)
            else:
                target_params = expert_trajectory[start_epoch + args.expert_epochs]

            target_params = torch.cat(
                [p.data.to(self.device).reshape(-1) for p in target_params], 0
            )
            target_params_4_clom = torch.cat(
                [p.data.to(self.device).reshape(-1) for p in expert_trajectory[-1]], 0
            )

            params_dict = dict(model_4_clom.named_parameters())

            offset = 0
            for (name, param) in params_dict.items():
                numel = param.numel()
                param.data.copy_(target_params_4_clom[offset: offset + numel].view_as(param))
                offset += numel
                    
            # model_4_clom.load_state_dict(params_dict)
            
            for param in model_4_clom.parameters():
                param.requires_grad = False

            # 初始化学生参数
            student_params = [
                torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params],0,).requires_grad_(True)
            ]

            starting_params = torch.cat([p.data.to(self.device).reshape(-1) for p in starting_params],0)

            param_loss_list = []
            param_dist_list = []

            logging.info(
                "it:{}--feat_max = {:.4f}, feat_min = {:.4f}".format(
                    it, torch.max(self.feat_syn), torch.min(self.feat_syn)
                )
            )

            feat_syn = self.feat_syn  # [n_nodes, feat_dim]

            # 构造自环 edge_index
            num_nodes = feat_syn.shape[0]
            edge_index_syn = torch.arange(num_nodes, device=feat_syn.device)
            edge_index_syn = torch.stack([edge_index_syn, edge_index_syn], dim=0)  # shape: [2, num_nodes]

            # 构造用于模型 forward 的 Data 对象
            if args.method == 'HGNN':
                # 对于HGNN模型，使用单位矩阵作为G矩阵
                G_matrix = torch.eye(num_nodes).to(feat_syn.device)
                data_syn = Data(x=feat_syn.clone(), edge_index=G_matrix).to(feat_syn.device)
            else:
                # 对于其他模型，使用标准的edge_index
                data_syn = Data(x=feat_syn.clone(), edge_index=edge_index_syn.clone()).to(feat_syn.device)
            
            data_syn = add_norm_to_data(data_syn)

            # tag
            for step in range(args.syn_steps):
                # 将 flat_param 拷贝到模型参数中
                forward_params = student_params[-1]

                # 前向传播获取输出
                output = model.forward(data_syn, flat_param=forward_params)
                if isinstance(output, tuple):
                    output_syn = output[0]  # 只要 logits 部分
                else:
                    output_syn = output

                if args.soft_label:
                    output_syn = F.log_softmax(output_syn, dim=1)
                    labels_syn = F.log_softmax(self.labels_syn, dim=1)
                    loss_syn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)(
                        output_syn, labels_syn
                    )
                    acc_syn = utils.accuracy(output_syn, torch.argmax(self.labels_syn, dim=1))
                else:
                    # 使用 CrossEntropyLoss 以避免 log-softmax 不匹配问题
                    loss_syn = F.cross_entropy(output_syn, self.labels_syn)
                    acc_syn = utils.accuracy(output_syn, self.labels_syn)

                # 计算梯度并更新参数
                grad = torch.autograd.grad(loss_syn, student_params[-1], create_graph=True)[0]
                student_params[-1] = student_params[-1] - self.syn_lr * grad

                if step % 500 == 0:
                    test_param = student_params[-1]

                    output_test = model.forward(self.data_all, flat_param=test_param)
                    if isinstance(output_test, tuple):
                        output_test = output_test[0]

                    # 评估准确率
                    acc_test = utils.accuracy(
                        output_test[self.test_idx],
                        self.data_all.y[self.test_idx]
                    )

                    logging.info(
                        "loss = {:.4f}, acc_syn = {:.4f}, acc_test = {:.4f}".format(
                            loss_syn.item(), acc_syn.item(), acc_test.item()
                        )
                    )
            # 计算参数loss
            param_loss = torch.tensor(0.0).to(self.device)
            param_dist = torch.tensor(0.0).to(self.device)
            
            param_diff = student_params[-1] - target_params
            param_loss += torch.norm(param_diff, 2)
            param_dist += torch.norm(starting_params - target_params, 2)

            param_loss_list.append(param_loss)
            param_dist_list.append(param_dist)

            param_loss = param_loss / num_params
            param_dist = param_dist / num_params
            param_loss = param_loss / param_dist

            grand_loss = param_loss

            print("param_loss raw:", param_loss.item(), "param_dist:", param_dist.item())


            # 构造 PyG Data 对象
            if args.method == 'HGNN':
                # 对于HGNN模型，使用单位矩阵作为G矩阵
                G_matrix = torch.eye(num_nodes).to(feat_syn.device)
                data_clom = Data(x=feat_syn, edge_index=G_matrix).to(feat_syn.device)
            else:
                # 对于其他模型，使用标准的edge_index
                data_clom = Data(x=feat_syn, edge_index=edge_index_syn).to(feat_syn.device)
                
            # 添加标记，表明这是蒸馏中的简化图结构
            data_clom = add_norm_to_data(data_clom)
            # 前向传播
            output_clom = model_4_clom.forward(data_clom, flat_param=target_params_4_clom)
            if isinstance(output_clom, tuple):
                output_clom = output_clom[0]

            if args.soft_label:
                output_clom = F.log_softmax(output_clom, dim=1)
                labels_syn = F.log_softmax(self.labels_syn, dim=1)
                loss_clom = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)(
                    output_clom, labels_syn
                )
            else:
                # 同样将比较 teacher（clom）输出与硬标签的损失改为 CrossEntropyLoss
                loss_clom = F.cross_entropy(output_clom, self.labels_syn)

            # 总loss
            total_loss = grand_loss + args.beta * loss_clom
       
            self.optimizer_feat.zero_grad()
            if args.soft_label:
                self.optimizer_label.zero_grad()
            if args.optim_lr:
                optimizer_lr.zero_grad()

            total_loss.backward()

            # ---------- Early-stopping for distill ----------
            if grand_loss < best_loss - 1e-4:        # 有明显下降
                best_loss  = grand_loss.item()
                best_loss_it = it
                wait_outer = 0
            else:
                wait_outer += 1
                if wait_outer >= patience_outer:
                    logging.info(f'Early-stop distill() at it {it}; '
                                f'best_loss={best_loss:.4f} (iter {best_loss_it})')
                    break
            # -----------------------------------------------

            self.optimizer_feat.step()
            if args.soft_label:
                self.optimizer_label.step()
            logging.info(
                "torch.sum(self.feat_syn) = {}".format(torch.sum(self.feat_syn))
            )
            if args.optim_lr:
                optimizer_lr.step()
                writer.add_scalar("student_lr_change", self.syn_lr.item(), it)

            # 检查NaN值
            if torch.isnan(total_loss) or torch.isnan(grand_loss):
                break  # 遇到NaN就退出
            if it % 1 == 0:
                import wandb
                if args.wandb:
                    wandb.log({"total_loss": total_loss.item()})
                logging.info(
                    "Iteration {}: Total_Loss = {:.4f}, Grand_Loss={:.4f}, Start_Epoch= {}, Student_LR = {:6f}".format(
                        it, total_loss.item(), grand_loss.item(), start_epoch, self.syn_lr.item()
                    )
                )

            # --- 定期评估 ---
            if it in eval_it_pool and it > 0:
                for model_eval in model_eval_pool:
                    logging.info(
                        "Evaluation: model_train = {}, model_eval = {}, iteration = {}".format(
                            args.method, model_eval, it
                        )
                    )

                    best_acc_eval, best_acc_test = self.eval_synset(args)
                    feat_syn_save, H_syn_save, label_syn_save = self.synset_save()

                    best_acc_eval = np.mean(np.array(best_acc_eval))
                    best_acc_test = np.mean(np.array(best_acc_test))

                    if args.wandb:
                        wandb.log({"test_acc_mean": best_acc_test})

                    # 保存最优模型（ 保存H_syn）
                    if best_acc_test > best_accs_test[model_eval]:
                        best_accs_test[model_eval] = best_acc_test
                        best_accs_test_iter[model_eval] = it
                        torch.save(
                            H_syn_save,
                            f"{args.save_dir}/H_{args.dname}_{args.reduction_rate}_{args.seed}.pt",
                        )
                        torch.save(
                            feat_syn_save,
                            f"{args.save_dir}/feat_{args.dname}_{args.reduction_rate}_{args.seed}.pt",
                        )
                        torch.save(
                            label_syn_save,
                            f"{args.save_dir}/label_{args.dname}_{args.reduction_rate}_{args.seed}.pt",
                        )

                        logging.info(
                            "new best test_acc occurs: eval_acc = {:.4f}, test_acc = {:.4f}, iteration = {}".format(
                                best_acc_eval * 100.0, best_acc_test * 100.0, it
                            )
                        )

            # --- 每1000次 或最后一次 额外保存 ---
            if it % 1000 == 0 or it == args.ITER:
                feat_syn_save, H_syn_save, label_syn_save = self.synset_save()
                torch.save(H_syn_save, f"{args.save_dir}/H_{args.dname}_{args.reduction_rate}_{it}_{args.seed}.pt")
                torch.save(feat_syn_save, f"{args.save_dir}/feat_{args.dname}_{args.reduction_rate}_{it}_{args.seed}.pt")
                torch.save(label_syn_save, f"{args.save_dir}/label_{args.dname}_{args.reduction_rate}_{it}_{args.seed}.pt")

            # --- 更新最佳loss记录 ---
            if grand_loss.item() < best_loss:
                best_loss = grand_loss.item()
                best_loss_it = it

            writer.add_scalar("grand_loss_curve", grand_loss.item(), it)

            # 最后统一打印最好的结果
            for model_eval in model_eval_pool:
                logging.info(
                    "Evaluation ACC: {} best test_acc = {:.5f}, best_iter = {}".format(
                        model_eval, best_accs_test[model_eval], best_accs_test_iter[model_eval]
                    )
                )

            logging.info(
                "Smallest loss = {:.06f} found at iteration {}".format(
                    best_loss, best_loss_it
                )
            )


    def get_coreset_init(self, features, edge_index, labels, args, run):
        """
        获取核心集初始化 (图结构版本)

        Args:
            features: 原始节点特征矩阵 [N, F]
            edge_index: 边连接关系 [2, E]
            labels: 原始标签 [N]
            args: 参数对象
            run: 第几次重复（用于加载核心集索引）

        Returns:
            feat_train: 子集特征 [N', F]
            edge_index_sub: 子图边 [2, E']
            labels_train: 子集标签 [N']
        """
        logging.info('Loading from: {}'.format(self.args.coreset_init_path))
        idx_selected_train = np.load(
            f'{self.args.coreset_init_path}/'
            f'{self.args.dname}-reduce_{self.args.reduction_rate}-{self.args.core_method}/'
            f'idx_{self.args.dname}_{self.args.reduction_rate}_{self.args.core_method}_{self.args.coreset_seed}_{run-1}.npy'
        )

        # 选择核心节点的特征和标签
        feat_train = features[idx_selected_train]
        labels_train = labels[idx_selected_train]

        # 保留核心子集相关的边
        idx_selected_set = set(idx_selected_train.tolist())

        # 构建旧节点ID → 新节点ID映射（方便更新 edge_index）
        idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(idx_selected_train.tolist())}

        edge_index_sub = []

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in idx_selected_set and dst in idx_selected_set:
                edge_index_sub.append([idx_mapping[src], idx_mapping[dst]])

        if len(edge_index_sub) == 0:
            print("Warning: no internal edges. Adding self-loops instead.")
            edge_index_sub = torch.stack(
                [torch.arange(len(idx_selected_train)), torch.arange(len(idx_selected_train))],
                dim=0
            )
        else:
            edge_index_sub = torch.tensor(edge_index_sub, dtype=torch.long).t().contiguous()


        return feat_train, edge_index_sub, labels_train



    def generate_labels_syn(self, data):
        """
        生成合成标签 (无变化)
        
        Args:
            data: 数据集对象
            
        Returns:
            list: 合成标签列表
        """
        from collections import Counter

        counter = Counter(data.labels_train)
        num_class_dict = {}

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
            sum_ += num_class_dict[c]
            self.syn_class_indices[c] = [
                len(labels_syn),
                len(labels_syn) + num_class_dict[c],
            ]
            labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn




def one_hot(x, num_classes, center=True, dtype=np.float32):
    """
    将标签转换为one-hot编码
    
    Args:
        x: 输入标签
        num_classes: 类别数
        center: 是否居中
        dtype: 数据类型
        
    Returns:
        ndarray: one-hot编码的标签
    """
    assert len(x.shape) == 1
    one_hot_vectors = np.array(x[:, None] == np.arange(num_classes), dtype)
    if center:
        one_hot_vectors = one_hot_vectors - 1.0 / num_classes
    return one_hot_vectors


def calc(gntk, feat1, feat2, diag1, diag2, A1, A2):
    """
    计算GNTK(Graph Neural Tangent Kernel)
    
    Args:
        gntk: GNTK对象
        feat1: 特征1
        feat2: 特征2
        diag1: 对角线1
        diag2: 对角线2
        A1: 邻接矩阵1
        A2: 邻接矩阵2
        
    Returns:
        tensor: GNTK计算结果
    """
    return gntk.gntk(feat1, feat2, diag1, diag2, A1, A2)


def loss_acc_fn_train(data, k_ss, k_ts, y_support, y_target, reg=5e-2):
    """
    训练损失和准确率计算函数
    
    Args:
        data: 数据集对象
        k_ss: 支持集核矩阵
        k_ts: 目标集核矩阵
        y_support: 支持集标签
        y_target: 目标集标签
        reg: 正则化参数
        
    Returns:
        tuple: (mse_loss, acc)
            - mse_loss: MSE损失
            - acc: 准确率
    """
    # print(k_ss.device, torch.abs(torch.tensor(reg)).to(k_ss.device),torch.trace(k_ss).device, torch.eye(k_ss.shape[0]).device)
    k_ss_reg = (
        k_ss
        + torch.abs(torch.tensor(reg)).to(k_ss.device)
        * torch.trace(k_ss).to(k_ss.device)
        * torch.eye(k_ss.shape[0]).to(k_ss.device)
        / k_ss.shape[0]
    )
    
    # 计算预测值
    pred = torch.matmul(
        k_ts[data.idx_train, :].cuda(),
        torch.matmul(
            torch.linalg.inv(k_ss_reg).cuda(),
            torch.from_numpy(y_support).to(torch.float64).cuda(),
        ),
    )
    
    # 计算MSE损失
    mse_loss = torch.nn.functional.mse_loss(
        pred.to(torch.float64).cuda(),
        torch.from_numpy(y_target).to(torch.float64).cuda(),
        reduction="mean",
    )
    acc = 0
    return mse_loss, acc


def loss_acc_fn_eval(data, k_ss, k_ts, y_support, y_target, reg=5e-2):
    """
    评估损失和准确率计算函数
    
    Args:
        data: 数据集对象
        k_ss: 支持集核矩阵
        k_ts: 目标集核矩阵
        y_support: 支持集标签
        y_target: 目标集标签
        reg: 正则化参数
        
    Returns:
        tuple: (mse_loss, acc)
            - mse_loss: MSE损失
            - acc: 准确率
    """
    # 添加正则化项
    k_ss_reg = (
        k_ss + np.abs(reg) * np.trace(k_ss) * np.eye(k_ss.shape[0]) / k_ss.shape[0]
    )   
    # 计算预测值
    pred = np.dot(k_ts, np.linalg.inv(k_ss_reg).dot(y_support))   
    # 计算MSE损失和准确率
    mse_loss = 0.5 * np.mean((pred - y_target) ** 2)
    acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y_target, axis=1))
    return mse_loss, acc

def add_norm_to_data(data):
    # 检查edge_index的类型和形状
    if not hasattr(data, 'edge_index'):
        # 如果数据中没有edge_index，直接返回
        return data
    
    # 先克隆edge_index以防止原地修改
    if hasattr(data.edge_index, 'clone'):
        edge_index = data.edge_index.clone().detach()
    else:
        edge_index = data.edge_index
        
    num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x.size(0)
    
    # 检查edge_index是矩阵H还是标准边索引
    if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 and edge_index.size(0) == 2:
        # 标准边索引格式 [2, num_edges]
        row, col = edge_index
        # 统计每个节点的度数（它出现在多少个超边中）
        deg = degree(row, num_nodes=num_nodes, dtype=torch.float32)  # row 是节点 → 给每个节点算度
        norm_per_node = deg.pow(-0.5)
        norm_per_node[torch.isinf(norm_per_node)] = 0.0
        
        # 对 edge_index 的每一条边，从 row 获取对应的 norm 值，构成边级别 norm 向量
        edge_norm = norm_per_node[row].clone()  # 使用clone创建副本，防止原地修改
        
        # 添加归一化属性（不修改原始数据）
        data.norm = edge_norm
    else:
        # 对于HGNN等模型，edge_index可能已经是矩阵H或者其他格式
        # 这些模型在dataloader.py中自己处理了normalization，不需要在这里计算
        pass
    
    return data
