import sklearn.decomposition as sd
from sklearn.utils.extmath import randomized_svd
import pandas as pd

from utils.util import *


def read_data(dataset):
    print('Reading data')
    # TODO: change the path of the folder
    data = pd.read_csv('./data/%s.csv' % dataset, header=None)
    num_element = data[0][0]
    data = data[1:].reset_index(drop=True)
    return data, int(num_element)


def generate_reduction_matrix(sm, rd):
    print('Dimension reduction', rd)
    svd = sd.TruncatedSVD(n_components=rd)
    new_matrix = svd.fit_transform(sm)
    return new_matrix


def generate_decomposition_matrix(sm, rd, svd_model=True, seed=0):
    if svd_model:
        # Truncated svd
        print('SVD decomposition', rd)
        svd = sd.TruncatedSVD(n_components=rd, random_state=seed)
        k_emb = svd.fit_transform(sm)
        o_emb_T = svd.components_
        # print(svd.explained_variance_.sum())
    else:
        # None-negative matrix factorization
        print('Non-SVD decomposition', rd)
        nmf = sd.NMF(n_components=rd, random_state=seed)
        k_emb = nmf.fit_transform(sm)
        o_emb_T = nmf.components_
        # print(nmf.reconstruction_err_)
    return k_emb, o_emb_T.transpose()


def calculate_jacaard_similarity(ol1, ol2):
    intersection_list = set(ol1).intersection(set(ol2))
    union_list = set(ol1).union(set(ol2))

    return len(intersection_list) / len(union_list)


def generate_samples(sm, vector=None):
    samples = {}
    data = []
    comp1 = []
    comp2 = []
    # generate (positive/negative) samples based on Jaccard similarity
    for key in range(sm.shape[0]):
        print(key)
        # get the list of object containing this keyword
        row = sm.getrow(key)
        object_list = row.nonzero()[1].tolist()
        list_copy = object_list.copy()
    
        sampled_keys_connected = set()
        # get the keyword array of the sampled object
        key_nums_connected = 2
        while len(sampled_keys_connected) < key_nums_connected:
            # choose one object connected by this keyword
            object_connected = random.choice(list_copy)
            key_list_connected = sm.getcol(object_connected).nonzero()[0].tolist()
            random.shuffle(key_list_connected)
            sampled_keys_connected.update(key_list_connected)
            if key in sampled_keys_connected:
                sampled_keys_connected.remove(key)
            list_copy.remove(object_connected)
            if not list_copy:
                break
    
        # get negative samples from the unconnected object
        # sampled_keys_not_connected = set()
        # zero_idxs = np.where(row.todense().A1 == 0)[0].tolist()
        # key_nums_not_connected = 2
        # while len(sampled_keys_not_connected) < key_nums_not_connected:
        #     object_not_connected = random.choice(zero_idxs)
        #     key_list_not_connected = sm.getcol(object_not_connected).nonzero()[0].tolist()
        #     random.shuffle(key_list_not_connected)
        #     sampled_keys_not_connected.update(key_list_not_connected)
        #     zero_idxs.remove(object_not_connected)

        if len(sampled_keys_connected) < 2:
            continue

        data.append(vector[key])
        sample_result = {}
        sampled_keys = list(sampled_keys_connected)[:2]
        # sampled_keys_connected.union(sampled_keys_not_connected)
        new_object_list1 = sm.getrow(sampled_keys[0]).nonzero()[1].tolist()
        sim1 = calculate_jacaard_similarity(object_list, new_object_list1)
        new_object_list2 = sm.getrow(sampled_keys[1]).nonzero()[1].tolist()
        sim2 = calculate_jacaard_similarity(object_list, new_object_list2)
        # for sk in sampled_keys:
        #     new_object_list = sm.getrow(sk).nonzero()[1].tolist()
        #     sim = calculate_jacaard_similarity(object_list, new_object_list)
        #     sample_result[sk] = sim
        if sim1 >= sim2:
            sort_sampled_keys = sampled_keys
            comp1.append(vector[sampled_keys[0]])
            comp2.append(vector[sampled_keys[1]])
        else:
            sort_sampled_keys = [sampled_keys[1], sampled_keys[0]]
            comp2.append(vector[sampled_keys[0]])
            comp1.append(vector[sampled_keys[1]])
        samples[key] = sort_sampled_keys
    
    return samples, np.array(data), np.array(comp1), np.array(comp2)