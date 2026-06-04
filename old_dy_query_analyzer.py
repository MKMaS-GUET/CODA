import time
from pathlib import Path

from utils.preprocess import *
from utils.criterion import *
from utils.customized_dataset import *
from utils.arg_parser import *
from model.encoder import *
from model.ace import *
if __name__ == '__main__':
    args = make_args()
    dataset = args.d
    dim = args.dim
    distill_ratio = args.r
    workload_type = args.qt
    workload_freq = args.qf
    type = workload_type
    lr = args.ql
    reg_weight = args.qr
    batch_size = 10
    if args.dev < 0:
        dev = 'cpu'
    else:
        dev = 'cuda:%d' % args.dev
    print(args)

    seed_random(args.s)
    _, num_element = read_data(dataset)
    element_emb = nn.Embedding(num_element + 1, dim, padding_idx=0).requires_grad_(False)

    # TODO: change the path of the folder
    folder = os.listdir('/opt/data/private/ACE/data/dynamic/%s' % dataset)
    num_parts = len(folder) - 1
    train_times = []
    test_mean = []
    test_median = []
    test_95_quan = []
    test_99_quan = []
    test_max_quan = []

    for idx in range(num_parts):
        group_embs = torch.load('save_model/%s/groups_%d.pth' % (dataset, idx))
        group_embs = group_embs.detach().to(dev).requires_grad_(False)
        degree = np.load('%s_freq_%d.npy' % (dataset, idx))
        degree = torch.from_numpy(degree)
        estimator = ACE(dim, args.cross_attn_num, args.self_attn_num, args.mlp_dep, args.mlp_dim).to(dev)
        if Path('save_model/%s_estimator_%s.pth' % (dataset, workload_type)).exists():
            estimator.load_state_dict(torch.load('save_model/%s_estimator_%s.pth' % (dataset, workload_type)))
        optimizer = torch.optim.Adam([{'params': estimator.parameters(), 'lr': lr}])#, {'params': group_embs, 'lr': lr}, 'weight_decay': 1e-3
        regularizer = ModelRegularizer(reg_weight)

        if (idx+1) < num_parts * 0.9:
            # training
            train_workload = pd.read_csv('/opt/data/private/ACE/query/%s/dynamic/%s/train/%s.csv' % (dataset, type, idx), header=None)
            val_workload = pd.read_csv('/opt/data/private/ACE/query/%s/dynamic/%s/val/%s.csv' % (dataset, type, idx), header=None)
            train_dataset = MyQueryDataset(train_workload)
            train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=MyQueryDataset.collate_fn)
            val_dataset = MyQueryDataset(val_workload)
            val_loader = DataLoader(val_dataset, shuffle=True, collate_fn=MyQueryDataset.collate_fn)

            train_start = time.perf_counter()
            best_loss = 1e50
            best_epoch = 0
            for epoch in range(2):
                for _, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    qs, truth = batch
                    truth = truth.float().to(dev)
                    lst = [element_emb(q + 1) for q in qs]
                    sel = [torch.log(degree[q]) for q in qs]
                    sel = pad_sequence(sel, True).unsqueeze(-1).to(dev)
                    lst = pad_sequence(lst, True).to(dev)

                    pred,_= estimator(lst, group_embs, sel)

                    l_train = weight_q_error(truth, pred).sum()
                    l_reg = regularizer(estimator)
                    train_loss = l_train + l_reg
                    train_loss.backward()
                    optimizer.step()
                
                if epoch >= 0:
                    with torch.no_grad():
                        val_preds = torch.tensor([])
                        val_truth = torch.tensor([])
                        for _, val_batch in enumerate(val_loader):
                            val_qs, val_cs = val_batch
                            val_truth = torch.cat((val_truth, val_cs))
                            val_lst = [element_emb(q + 1) for q in val_qs]
                            val_sel = [torch.log(degree[q]) for q in val_qs]
                            val_sel = pad_sequence(val_sel, True).unsqueeze(-1).to(dev)
                            val_lst = pad_sequence(val_lst, True).to(dev)

                            val_pred,_= estimator(val_lst, group_embs, val_sel)

                            val_preds = torch.cat((val_preds, val_pred.cpu()))
                        val_loss = q_error_criterion(val_truth, val_preds).mean()
                        if val_loss.item() < best_loss and val_preds.min().item() > 0:
                            best_loss = val_loss.item()
                            best_epoch = epoch
                            torch.save(estimator.state_dict(), 'save_model/%s_estimator_%s.pth' % (dataset, workload_type))
                    torch.cuda.empty_cache()
            train_end = time.perf_counter()
            train_times.append((train_end - train_start) * 1000)
        else:
            # testing
            test_workload = pd.read_csv('/opt/data/private/ACE/query/%s/dynamic/%s/test_%s/%s.csv' % (dataset, workload_type, workload_freq, idx), header=None)
            test_dataset = MyQueryDataset(test_workload)
            test_loader = DataLoader(test_dataset, batch_size, collate_fn=MyQueryDataset.collate_fn)
            with torch.no_grad():   
                test_preds = torch.tensor([])
                test_truth = torch.tensor([])
                for _, test_batch in enumerate(test_loader):
                    test_qs, test_cs = test_batch
                    test_truth = torch.cat((test_truth, test_cs))
                    test_lst = [element_emb(q + 1) for q in test_qs]
                    test_sel = [torch.log(degree[q]) for q in test_qs]
                    test_sel = pad_sequence(test_sel, True).unsqueeze(-1).to(dev)

                    test_lst = pad_sequence(test_lst, True).to(dev)
                    test_pred,_= estimator(test_lst, group_embs, test_sel)

                    test_preds = torch.cat((test_preds, test_pred.cpu()))
            test_loss = q_error_criterion(test_truth, test_preds)
            test_mean.append(test_loss.mean().item())
            test_median.append(test_loss.median().item())
            test_95_quan.append(test_loss.quantile(0.95, dim=0, interpolation='nearest').item())
            test_99_quan.append(test_loss.quantile(0.99, dim=0, interpolation='nearest').item())
            test_max_quan.append(test_loss.quantile(1.00, dim=0, interpolation='nearest').item())
        
    print('Train', np.mean(train_times) / 1000)
    print("%.3f, %.3f, %.3f, %.3f, %.3f" % (np.mean(test_mean), np.mean(test_median), np.mean(test_95_quan), np.mean(test_99_quan),np.mean(test_max_quan)))
