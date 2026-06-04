from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()

    parser.add_argument('--d', type=str, help='the dataset', default='wiki')
    parser.add_argument('--dev', type=int, help='device', default=0)
    parser.add_argument('--s', type=int, help='seed', default=0)
    parser.add_argument('--m', type=str, help='train model or test model', default='train')

    # ----------------------------------- Model Structure Params -----------------------------------
    parser.add_argument('--dim', type=int, help='feature dimension', default=64)
    parser.add_argument('--dis_dep', type=int, help='the depth of distillation model', default=4)
    parser.add_argument('--cross_attn_num', type=int, help='the depth of cross attention (analyzer)', default=4)
    parser.add_argument('--self_attn_num', type=int, help='the depth of self attention (analyzer)', default=8)
    parser.add_argument('--mlp_dep', type=int, help='the depth of mlp', default=6)
    parser.add_argument('--mlp_dim', type=int, help='the hidden dimension of mlp', default=512)

    # ----------------------------------- Data Featurization Params -----------------------------------
    parser.add_argument('--de', type=int, help='number of epochs (featurization)', default=1)
    parser.add_argument('--neg', type=int, help='number of negative samples', default=10)
    parser.add_argument('--db', type=int, help='batch size of data featurization', default=10000)
    parser.add_argument('--r', type=float, help='distillation ratio', default=1e-3)
    parser.add_argument('--dl', type=float, help='init learning rate', default=1e-3)
    parser.add_argument('--dr', type=float, help='regularization weight', default=1e-4)

    # ----------------------------------- Query Analyzer Params -----------------------------------
    parser.add_argument('--ql', type=float, help='init learning rate', default=1e-3)
    parser.add_argument('--qr', type=float, help='regularization weight', default=1e-4)
    parser.add_argument('--qb', type=int, help='batch size of query analyzer', default=100)
    parser.add_argument('--qe', type=int, help='number of epochs (analyzer)', default=100)
    parser.add_argument('--qt', type=str, help='workload type', default='subset')
    parser.add_argument('--qf', type=str, help='workload frequency', default='regular')
    parser.add_argument('--temp', type=float, default=0.8, help='contrastive loss temperature')
    parser.add_argument('--q_noise', type=float, default=0.0, help='query noise scale')
        # ======= 新增 Flow 相关参数 =======


    parser.add_argument('--cl_weight', type=float, default=0.1, help='对比损失权重')
    parser.add_argument('--temperature', type=float, default=0.07, help='对比损失温度')
    parser.add_argument('--proj_dim', type=int, default=128, help='投影头输出维度')
    parser.add_argument('--pos_jac', type=float, default=0.5, help='正样本Jaccard阈值')
    parser.add_argument('--pos_label', type=float, default=0.2, help='正样本基数阈值')

    args = parser.parse_args()

    return args
