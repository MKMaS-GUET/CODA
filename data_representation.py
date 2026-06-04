import time
from pathlib import Path

from utils.preprocess import *
from utils.customized_dataset import *
from utils.arg_parser import *
from model.encoder import *
from model.ace import *


if __name__ == '__main__':
    args = make_args()

    dataset = args.d
    dim = args.dim
    distill_ratio = args.r
    if args.dev < 0:
        dev = 'cpu'
    else:
        dev = 'cuda:%d' % args.dev
    print(args)

    seed_random(args.s)
    data, num_element = read_data(dataset)
    element_emb = nn.Embedding(num_element + 1, dim, padding_idx=0).requires_grad_(False)

    agg_model = Aggregator(dim)
    agg_model.load_state_dict(torch.load('save_model/%s_aggregation.pth' % (dataset)))
    dis_model = PostDistillation(args.dis_dep, dim, dim, False)
    dis_model.load_state_dict(torch.load('save_model/%s_distillation.pth' % (dataset)))

    # if not Path('save_model/%s_groups_%s_%s.pth' % (dataset, dim, distill_ratio)).exists():
    if not Path('save_model/%s_groups.pth' % (dataset)).exists():
        data_set = SequentialDataset(data, 1, args.neg, num_element)
        loader = DataLoader(data_set, args.db, collate_fn=SequentialDataset.collate_fn)
        print(len(loader))
        start = time.time()
        group_embs = torch.tensor([])
        # data featurization: use the pre-trained model to get data featurization (group embedding)
        print('Featurization in progress...')
        with torch.no_grad():
            for _, (data, pos_idx, neg_idx) in enumerate(loader):
                s = time.perf_counter()
                data = pad_sequence(data, True)
                element_embs = element_emb(data)
                masks = torch.sign(torch.sum(torch.abs(element_embs), -1))
                masks = masks.unsqueeze(-1)
                masks = masks.expand(-1, -1, element_embs.size(-1))
                pos_sample = element_emb(pos_idx)
                neg_sample = element_emb(neg_idx)
                
                set_embs, _= agg_model(element_embs, masks)

                group_idx = torch.randperm(set_embs.size(0))[:max(int(set_embs.size(0) * distill_ratio), 2)]
                groups = set_embs[group_idx].detach()
                groups = dis_model(set_embs.unsqueeze(0), groups)
                group_embs = torch.cat((group_embs, groups))
        # torch.save(group_embs, 'save_model/%s_groups_%s_%s.pth' % (dataset, dim, distill_ratio))
        torch.save(group_embs, 'save_model/%s_groups.pth' % (dataset))
        end = time.time()
        print(end - start)
