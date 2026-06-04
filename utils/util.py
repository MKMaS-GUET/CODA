import random
import os
import torch
import numpy as np
import pandas as pd
import scipy.sparse as ss

def seed_random(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # using multiple GPUs
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def generate_sparse_matrix(d):
    print('Generate sparse matrix')
    non_zero_nums = [0]
    non_zero_column_index = []
    key_arr = []
    obj_arr = []
    for i in range(d.shape[0]):
        terms = d[0][i].strip().split(' ')
        non_zero_nums.append(non_zero_nums[-1] + len(terms))
        for term in terms:
            non_zero_column_index.append(int(term))
            key_arr.append(int(term))
            obj_arr.append(i)
    non_zero_values = [1] * len(non_zero_column_index)
    csr = ss.csr_matrix((non_zero_values, non_zero_column_index, non_zero_nums))
    edge_arr = np.array([key_arr, list(np.array(obj_arr) + csr.shape[1])])
    return csr.transpose(), edge_arr


def generate_bipartite_sparse_matrix(d):
    print('Generate bipartite sparse matrix')
    non_zero_nums = [0]
    non_zero_column_index = []
    key_arr = []
    obj_arr = []
    for i in range(d.shape[0]):
        terms = d[0][i].strip().split(' ')
        non_zero_nums.append(non_zero_nums[-1] + len(terms))
        for term in terms:
            non_zero_column_index.append(int(term))
            key_arr.append(int(term))
            obj_arr.append(i)
    non_zero_values = [1] * len(non_zero_column_index)
    csr = ss.csr_matrix((non_zero_values, non_zero_column_index, non_zero_nums))
    edge_arr = np.array([key_arr, obj_arr])
    return csr.transpose(), edge_arr


def or_find(q, g):
# OR operation
    for x in q:
        if str(x) in g:
            return True
    return False


def calculate_cardinality(query, data):
    res = data[data[0].apply(lambda x: or_find(query, x))]
    return res.shape[0]


def generate_embedding(input, r_drop=0.5):
    # drop out
    # print(input)
    dropout = torch.nn.Dropout(p=r_drop)
    out = dropout(input)
    # print(out)
    # pooling
    # pooling = torch.nn.MaxPool1d(input.shape[1])
    pooling = torch.nn.AvgPool1d(input.shape[1])
    res = pooling(out)
    return res.numpy().transpose()


def prepare_data_label(df, vector, mapping=None, degree=None):
    df[0] = df[0].astype(str)
    new_df = pd.concat([df[0].str.split(' ', expand=True), df[1]], axis=1)
    new_df.columns = range(new_df.shape[1])
    label_df = new_df.loc[:, new_df.shape[1] - 1:]
    l = label_df.to_numpy()
    l = np.expand_dims(l, axis=1)
    l[l == 0] = 1
    # w = l / np.sum(l)
    w = np.log(l) / np.sum(np.log(l))
    # w = l / np.mean(l)
    # w = np.clip(w, 1e-3, np.max(w))
    l = torch.from_numpy(l).float()
    w = torch.from_numpy(w).float()
    data_df = new_df.loc[:, 0:new_df.shape[1] - 2].astype(int)
    if mapping is not None:
        data_df.replace(mapping, inplace=True)
    idx = data_df.to_numpy()
    d = vector[idx]
    if degree is not None:
        d = torch.cat((d, degree[idx]), -1)

    return d, l, w


def prepare_data(df, mapping=None):
    df[0] = df[0].astype(str)
    new_df = pd.concat([df[0].str.split(' ', expand=True), df[1]], axis=1)
    new_df.columns = range(new_df.shape[1])
    label_df = new_df.loc[:, new_df.shape[1] - 1:]
    l = label_df.to_numpy()
    l = np.expand_dims(l, axis=1)
    l[l == 0] = 1
    # w = l / np.sum(l)
    w = np.log(l) / np.sum(np.log(l))
    # w = l / np.mean(l)
    # w = np.clip(w, 1e-3, np.max(w))
    l = torch.from_numpy(l).float()
    w = torch.from_numpy(w).float()
    data_df = new_df.loc[:, 0:new_df.shape[1] - 2].astype(int)
    if mapping is not None:
        data_df.replace(mapping, inplace=True)
    idx = torch.from_numpy(data_df.to_numpy())

    return idx, l, w


def prepare_variable_data(df):
    df[0] = df[0].astype(str)
    new_df = pd.concat([df[0].str.split(' ', expand=False), df[1]], axis=1)
    new_df.columns = range(new_df.shape[1])
    label_df = new_df.iloc[:, 1:]
    l = label_df.to_numpy()
    l = np.expand_dims(l, axis=1)
    l[l == 0] = 1
    # w = l / np.sum(l)
    # w = np.log(l) / np.sum(np.log(l))
    # w = l / np.mean(l)
    # w = np.clip(w, 1e-3, np.max(w))
    l = torch.from_numpy(l).float()
    # w = torch.from_numpy(w).float()
    idx = new_df.loc[:, 0]
    
    return idx, l


def get_1_hop_neighbor(node, edge_index):
    index = torch.where(edge_index[0] == node)[0]
    res_node = edge_index[1, index]
    return res_node


def concatenate_embedding(emb1, emb2):
    return torch.cat((emb1, emb2), 0).float()


def convert_undirected_graph(edge_index):
    print('Convert to undirected graph')
    reverse_edge_index = edge_index[[1, 0]]
    edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
    print('edge_index', edge_index.shape)
    return edge_index


def negative_sampling(num_key, edge_array, idx, num_neg_sample, sm=None):
    print('Negative sampling')
    if sm is None:
        # 1. soft negative sampling: sample nodes from the rest nodes
        neg_idx_key = np.random.choice(np.setdiff1d(np.arange(0, num_key), np.array([edge_array[1][idx]])), size=(num_neg_sample, ))
    else:
        # 2. hard negative sampling: sample nodes from the unconnected nodes
        neg_idx_key = np.random.choice(np.setdiff1d(np.arange(0, num_key), sm[idx].nonzero()[1]), size=(num_neg_sample, ))
    return neg_idx_key
    
