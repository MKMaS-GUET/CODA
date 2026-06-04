
from torch.utils.data import random_split, Subset

from utils.preprocess import *
from utils.criterion import *
from utils.customized_dataset import *
from utils.arg_parser import *
from model.encoder import *


if __name__ == '__main__':
    args = make_args()

    dataset = args.d
    dim = args.dim
    num_neg_samples = args.neg
    batch_size = args.db
    distill_ratio = args.r
    distill_depth = args.dis_dep
    lr = args.dl
    reg_weight = args.dr
    if args.dev < 0:
        dev = 'cpu'
    else:
        dev = 'cuda:%d' % args.dev
    print(args)
    
    model_path = 'save_model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    seed_random(args.s)
    data, num_element = read_data(dataset)
    element_emb = nn.Embedding(num_element + 1, dim, padding_idx=0).requires_grad_(False)
    # torch.save(element_emb.state_dict(), '%s_elements_%s_%s.pth' % (dataset, dim, distill_ratio))
    # data = data[:int(data.shape[0] * 2 / 3)].reset_index(drop=True)
    
    data_set = SequentialDataset(data, 1, args.neg, num_element)

    train_size = int(len(data_set) * 0.8)
    test_size = len(data_set) - train_size

    # dynamic case
    # train_size = int(len(data_set) * 0.4)
    # test_size = int(len(data_set) * 0.1)

    # train_data = Subset(data_set, list(range(train_size)))
    # test_data = Subset(data_set, list(range(train_size, train_size + test_size)))

    train_data, test_data = random_split(data_set, [train_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size, True, collate_fn=SequentialDataset.collate_fn)
    test_loader = DataLoader(test_data, batch_size, True, collate_fn=SequentialDataset.collate_fn)
    print(len(train_loader), len(test_loader))
    # loader = DataLoader(data_set, batch_size, collate_fn=SequentialDataset.collate_fn)
    # print(len(loader))

    # model = Featurization(distill_depth, dim, dim, int(distill_ratio * batch_size), False).to(dev)
    # model = Featurization(distill_depth, dim, dim, distill_ratio, False).to(dev)
    agg_model = Aggregator(dim).to(dev)
    dis_model = PostDistillation(distill_depth, dim, dim, False).to(dev)
    optimizer = torch.optim.Adam([{'params': agg_model.parameters(), 'lr': lr}, {'params': dis_model.parameters(), 'lr': lr}])#, 'weight_decay': 2e-3
    link_loss = nn.CrossEntropyLoss()
    mmd_loss = MMD_Loss().to(dev)
    regularizer = ModelRegularizer(reg_weight)

    best_loss = 1e50
    best_epoch = 0

    # loss_file = open('temp%s_%s_loss.txt' % (distill_depth, distill_ratio), 'w')
    for epoch in range(args.de):
        for batch, (idx, pos_idx, neg_idx) in enumerate(train_loader):
            idx = pad_sequence(idx, True)
            element_embs = element_emb(idx).to(dev)
            masks = torch.sign(torch.sum(torch.abs(element_embs), -1))
            masks = masks.unsqueeze(-1)
            masks = masks.expand(-1, -1, dim)
            pos_sample = element_emb(pos_idx).to(dev)
            neg_sample = element_emb(neg_idx).to(dev)
            # groups = torch.randn(max(1, int(agg_embs.size(0) * distill_ratio)), dim).to(dev)

            optimizer.zero_grad()
            link_truth = torch.cat((torch.ones_like(pos_idx), torch.zeros_like(neg_idx)), dim=1).float().to(dev)
            # aggregation
            # set_embs = agg_model(element_embs, masks)
            set_embs, _ = agg_model(element_embs,masks)
            
            concat_samples = torch.cat((pos_sample, neg_sample), dim=1)
            link_pred = (set_embs.unsqueeze(1) * concat_samples).sum(-1)

            sample_idx = torch.randperm(set_embs.size(0))[:max(int(set_embs.size(0) * distill_ratio), 2)]
            distill_set_embs = set_embs[sample_idx].detach()
            # distillation
            distill_set_embs = dis_model(set_embs.unsqueeze(0), distill_set_embs)
            
            ce = link_loss(link_pred, link_truth)
            mmd = mmd_loss(set_embs, distill_set_embs)
            reg = regularizer(agg_model) + regularizer(dis_model)
            loss = ce + mmd + reg
            # print(loss.item(), mmd.item())
            # print('Train', loss.item(), ce_loss.item(), mmd_loss.item(), reg_loss.item())
            loss.backward()
            optimizer.step()
            # exit()

            if batch % 5 == 0:
                with torch.no_grad():
                    loss1 = torch.tensor([])
                    loss2 = torch.tensor([])
                    for _, (test_idx, test_pos_idx, test_neg_idx) in enumerate(test_loader):
                        test_idx = pad_sequence(test_idx, True)
                        test_elemnt_embs = element_emb(test_idx).to(dev)
                        mask = torch.sign(torch.sum(torch.abs(test_elemnt_embs), -1))
                        mask = mask.unsqueeze(-1)
                        mask = mask.expand(-1, -1, dim)
                        test_pos_sample = element_emb(test_pos_idx).to(dev)
                        test_neg_sample = element_emb(test_neg_idx).to(dev)

                        test_link_truth = torch.cat((torch.ones_like(test_pos_idx), torch.zeros_like(test_neg_idx)), dim=1).float()
                        # test_link_pred, test_sets, test_groups = model(test_agg_embs, test_pos_sample, test_neg_sample, mask)

                        test_set_embs, _ = agg_model(test_elemnt_embs, mask)
                        test_concat_samples = torch.cat((test_pos_sample, test_neg_sample), dim=1)
                        test_link_pred = (test_set_embs.unsqueeze(1) * test_concat_samples).sum(-1)

                        test_sample_idx = torch.randperm(test_set_embs.size(0))[:max(int(test_set_embs.size(0) * distill_ratio), 2)]
                        test_distill_set_embs = test_set_embs[test_sample_idx].detach()
                        test_distill_set_embs = dis_model(test_set_embs.unsqueeze(0), test_distill_set_embs)

                        test_ce_loss = link_loss(test_link_pred.cpu(), test_link_truth)
                        test_mmd_loss = mmd_loss(test_set_embs, test_distill_set_embs).cpu()
                        loss1 = torch.cat((loss1, test_ce_loss.reshape(1)))
                        loss2 = torch.cat((loss2, test_mmd_loss.reshape(1)))
                    # print(test_ce_loss, test_mmd_loss)
                    # print(batch)
                    # loss_file.write(str(loss2.mean().item()) + ' ' + str(loss2.max().item()) + '\n')
                    # loss_file.flush()
                    # print(epoch, loss1.max().item(), loss2.max().item(), (loss1 + loss2).max().item(), (loss1 + loss2).min().item())
                    if (loss1 + loss2).max().item() < best_loss:
                        # torch.save(agg_model.state_dict(), 'save_model/%s_aggregation_%s_%s_%s.pth' % (dataset, dim, distill_depth, distill_ratio))
                        # torch.save(dis_model.state_dict(), 'save_model/%s_distillation_%s_%s_%s.pth' % (dataset, dim, distill_depth, distill_ratio))
                        torch.save(agg_model.state_dict(), 'save_model/%s_aggregation.pth' % (dataset))
                        torch.save(dis_model.state_dict(), 'save_model/%s_distillation.pth' % (dataset))
                        best_loss = (loss1 + loss2).max().item()
                        best_epoch = batch
    
    print(best_epoch, best_loss)
    # loss_file.close()
