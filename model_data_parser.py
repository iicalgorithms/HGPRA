import argparse

def load_hnn_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--dname', default='citeseer',
                        choices=['cora', 'citeseer', 'pubmed', 'coauthor_cora', 'coauthor_dblp', 'NTU2012', 'ModelNet40', 'zoo', 'Mushroom', '20newsW100', 'yelp', 'house-committees-100', 'walmart-trips-100'])
    parser.add_argument('--method', default='AllDeepSets', 
                        choices=['AllDeepSets', 'AllSetTransformer', 'HyperGCN', 'HGNN', 'HNHN', 'MLP', 'UniGCNII', 'CEGCN', 'CEGAT', 'HCHA'])
    parser.add_argument('--epochs', default=200, type=int)
    # Number of runs for each split (test fix, only shuffle train/val)
    parser.add_argument('--runs', default=1, type=int)
    parser.add_argument('--cuda', default=0, choices=[-1, 0, 1], type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--wd', default=0.0, type=float)
    # How many layers of full NLConvs
    parser.add_argument('--All_num_layers', default=1, type=int)
    parser.add_argument('--MLP_num_layers', default=2, type=int)  # How many layers of encoder
    parser.add_argument('--MLP_hidden', default=128, type=int)  # Encoder hidden units
    parser.add_argument('--Classifier_num_layers', default=1,
                        type=int)  # How many layers of decoder
    parser.add_argument('--Classifier_hidden', default=128,
                        type=int)  # Decoder hidden units
    parser.add_argument('--display_step', type=int, default=1)
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    # ['all_one','deg_half_sym']
    parser.add_argument('--normtype', default='all_one')
    parser.add_argument('--add_self_loop', action='store_false')
    # NormLayer for MLP. ['bn','ln','None']
    parser.add_argument('--normalization', default='ln')
    parser.add_argument('--deepset_input_norm', default = True)
    parser.add_argument('--GPR', action='store_false')  # skip all but last dec
    # skip all but last dec
    parser.add_argument('--LearnMask', action='store_false')
    parser.add_argument('--num_features', default=0, type=int)  # Placeholder
    parser.add_argument('--num_classes', default=0, type=int)  # Placeholder
    # Choose std for synthetic feature noise
    parser.add_argument('--feature_noise', default='1', type=str)
    # whether the he contain self node or not
    parser.add_argument('--exclude_self', action='store_true')
    parser.add_argument('--PMA', action='store_true')
    #     Args for HyperGCN
    parser.add_argument('--HyperGCN_mediators', action='store_true')
    parser.add_argument('--HyperGCN_fast', action='store_true')
    #     Args for Attentions: GAT and SetGNN
    parser.add_argument('--heads', default=1, type=int)  # Placeholder
    parser.add_argument('--output_heads', default=1, type=int)  # Placeholder
    #     Args for HNHN
    parser.add_argument('--HNHN_alpha', default=-1.5, type=float)
    parser.add_argument('--HNHN_beta', default=-0.5, type=float)
    parser.add_argument('--HNHN_nonlinear_inbetween', default=True, type=bool)
    #     Args for HCHA
    parser.add_argument('--HCHA_symdegnorm', action='store_true')
    #     Args for UniGNN
    parser.add_argument('--UniGNN_use-norm', action="store_true", help='use norm in the final layer')
    parser.add_argument('--UniGNN_degV', default = 0)
    parser.add_argument('--attn_drop', default = 0.1)
    parser.add_argument('--UniGNN_degE', default = 0)
    
    parser.set_defaults(PMA=True)  # True: Use PMA. False: Use Deepsets.
    parser.set_defaults(add_self_loop=False)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(GPR=False)
    parser.set_defaults(LearnMask=False)
    parser.set_defaults(HyperGCN_mediators=True)
    parser.set_defaults(HyperGCN_fast=False)
    parser.set_defaults(HCHA_symdegnorm=False)

    return parser
