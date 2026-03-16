# from model import HGNN, HGNN_PLUS, HNHN, UNIGAT, HyperGCN


# def load_model(model_type, input_dim, hidden_dim, num_class, dropout=0.1, use_bn=False):
#     if model_type == 'HGNN':
#         return HGNN(input_dim, hidden_dim, num_class, num_layers=2, drop_rate=dropout)
#     elif model_type == 'HGNN_PLUS':
#         return HGNN_PLUS(input_dim, hidden_dim, num_class, num_layers=2, drop_rate=dropout)
#     elif model_type == 'HNHN':
#         return HNHN(input_dim, hidden_dim, num_class, num_layers=2, drop_rate=dropout)
#     elif model_type == 'UNIGAT':
#         return UNIGAT(input_dim, hidden_dim, num_class, num_layers=2, drop_rate=dropout)
#     elif model_type == 'HyperGCN':
#         return HyperGCN(input_dim, hidden_dim, num_class, num_layers=2, drop_rate=dropout)
#     else:
#         raise NotImplementedError(f"Model {model_type} not implemented.")

import torch_sparse
from models import *
from layers import *
from preprocessing import get_HyperGCN_He_dict
from scipy.sparse import coo_matrix
import torch
import torch.nn as nn
from dhg import Hypergraph


def parse_method(args, data):
    #     Currently we don't set hyperparameters w.r.t. different dataset
    if args.method == 'AllSetTransformer':
        args.aggregate = 'mean'
        # 使用SetGNNWrapper封装AllSetTransformer模型
        from model.setgnn_wrapper import SetGNNWrapper
        if args.LearnMask:
            model = SetGNNWrapper(args, data.norm)
        else:
            model = SetGNNWrapper(args)
    
    elif args.method == 'AllDeepSets':
        args.PMA = False
        args.aggregate = 'add'
        # 使用SetGNNWrapper封装AllDeepSets模型
        from model.setgnn_wrapper import SetGNNWrapper
        if args.LearnMask:
            model = SetGNNWrapper(args, data.norm)
        else:
            model = SetGNNWrapper(args)

#     elif args.method == 'SetGPRGNN':
#         model = SetGPRGNN(args)

    elif args.method == 'CEGCN':
        model = CEGCN(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'CEGAT':
        model = CEGAT(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      heads=args.heads,
                      output_heads=args.output_heads,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'HyperGCN':
        # 使用专门封装的 HyperGCNWrapper，不影响原有模型
        from model.hypergcn_wrapper import HyperGCNWrapper
        model = HyperGCNWrapper(
            in_channels=args.num_features,
            hid_channels=args.MLP_hidden,
            num_classes=args.num_classes,
            num_layers=args.All_num_layers,
            drop_rate=args.dropout
        )

    elif args.method == 'HGNN':
        # model = HGNN(in_ch=args.num_features,
        #              n_class=args.num_classes,
        #              n_hid=args.MLP_hidden,
        #              dropout=args.dropout)
        model = HGNN(in_ch=args.num_features,
                     n_class=args.num_classes,
                     n_hid=args.MLP_hidden,
                     dropout=args.dropout)

    elif args.method == 'HNHN':
        model = HNHN(args)

    elif args.method == 'HCHA':
        model = HCHA(args)

    elif args.method == 'MLP':
        model = MLP_model(args)
    elif args.method == 'UniGCNII' or args.method == 'UniGAT':
        if args.cuda in [0,1]:
            device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        # (row, col), value = torch_sparse.from_scipy(coo_matrix((torch.ones((data.edge_index.shape[1])).numpy(), (data.edge_index[0].cpu().numpy(), data.edge_index[1].cpu().numpy())), 
        #                                                        shape=(data.edge_index[0].max()+1, data.edge_index[1].max()+1)))
        V, E = data.edge_index[0, :], data.edge_index[1, :]

        V, E = V.to(device), E.to(device)
        
        if args.method == 'UniGCNII':
            model = UniGCNII(args, nfeat=args.num_features, nhid=args.MLP_hidden, nclass=args.num_classes, nlayer=args.All_num_layers, nhead=args.heads,
                            V=V, E=E)
        
        else:
            args.model_name = 'UniGAT'
            model = UniGNN(args, nfeat=args.num_features, nhid=args.MLP_hidden, nclass=args.num_classes, nlayer=args.All_num_layers, nhead=args.heads,
                           V=V, E=E)
        

    #     Below we can add different model, such as HyperGCN and so on
    return model
