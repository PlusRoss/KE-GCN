import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
import pickle as pkl
import os
import sys
from collections import Counter
import random

def align_relation(dataset):
    rel_list_0, rel_list_1, rel_list_2 = [], [], []
    assign_dict = defaultdict(list)
    file0 = dataset + "/triples_1"
    file1 = dataset + "/triples_2"
    file2 = "../../JAPE/data/dbp15k/{}/0_3/triples_2".format(dataset)
    with open(file0, "r") as rf0:
        for line in rf0.readlines():
            rel_list_0.append(line.strip().split("\t")[1])
    with open(file1, "r") as rf1:
        for line in rf1.readlines():
            rel_list_1.append(line.strip().split("\t")[1])
    with open(file2, "r") as rf2:
        for line in rf2.readlines():
            rel_list_2.append(line.strip().split("\t")[1])
    assert len(rel_list_1) == len(rel_list_2)
    shared_relation = set(rel_list_0).intersection(set(rel_list_1))
    print("original shared relation: ", len(shared_relation))
    for i in range(len(rel_list_1)):
        assign_dict[rel_list_2[i]].append(rel_list_1[i])
    aligned_relation_num = 0
    with open(dataset + "/ref_rel_ids", "w") as wf:
        for key in assign_dict:
            ind = list(set(assign_dict[key]))
            assert len(ind) == 1
            if key != ind[0]:
                aligned_relation_num += 1
                wf.write(key + "\t" + ind[0] + "\n")
    print("aligned relation: ", aligned_relation_num)
    # with open(file1, "r") as rf1:
    #     with open(dataset + "/triples_2_relaligned", "w") as wf:
    #         for i,line in enumerate(rf1.readlines()):
    #             line = line.split("\t")
    #             line[1] = rel_list_2[i]
    #             line = "\t".join(line)
    #             wf.write(line)


def process_wn():

    def one_hot(labels):
        label_num = np.max([np.max(i) for i in labels if len(i) > 0]) + 1
        label_onehot = np.zeros([len(labels), label_num])
        idx = []
        label_list = []
        for i, each in enumerate(labels):
            if len(each) > 0:
                idx.append(i)
                assert len(each) == 1
            for j in each:
                label_onehot[i][j] = 1.
                label_list.append(j)
        return label_onehot, idx, label_list

    train_data = np.load("class/wordnet/train_data.npz")
    test_data = np.load("class/wordnet/test_data.npz")
    print(train_data.files)
    labels = train_data["labels"]
    label_len_list = [len(i) for i in labels]
    print(labels)
    print("label num: {} dist: {}".format(len(labels), Counter(label_len_list)))
    y, idx, label_list = one_hot(labels)
    random.shuffle(idx)
    train = idx[:int(0.1*len(idx))]
    test = idx[int(0.1*len(idx)):]
    print("process node with label num: {} dist: {}".format(len(idx), Counter(label_list)))
    print("label num: ", len(Counter(label_list)))

    head_idx, rel_idx, tail_idx = [], [], []
    for tuple in train_data["train_data"]:
        head_idx.append(tuple[0])
        rel_idx.append(tuple[1])
        tail_idx.append(tuple[2])
    head_idx, rel_idx, tail_idx = set(head_idx), set(rel_idx), set(tail_idx)
    print("num of head id: {}, rel id: {}, tail id: {}".format(len(head_idx), len(rel_idx), len(tail_idx)))
    # print(len(head_idx.intersection(rel_idx)))
    # print(len(tail_idx.intersection(head_idx)))
    KG = np.concatenate([train_data["train_data"], test_data["test_data"]])
    print(KG.shape)
    e = np.max([train_data["nums_type"][0], train_data["nums_type"][2]])
    print(e)

    data = {'A': KG,
            'y': y,
            'train_idx': train,
            'test_idx': test,
            "e": e
            }
    with open('class/wordnetpro.pickle', 'wb') as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)



def process_fb():
    def filter(raw_list):
        num_dict = defaultdict(int)
        for item in raw_list:
            num_dict[item] += 1
        sort_dict = sorted([[key, num_dict[key]] for key in num_dict], key=lambda x:x[1])
        top_dict = sort_dict[-51:-1]
        # print(top_dict)
        return [each[0] for each in top_dict]

    def reorder(raw_list):
        order_dict = {}
        order = 0
        for item in raw_list:
            if item not in order_dict:
                order_dict[item] = order
                order += 1
        return order_dict

    data = {}
    KG = []
    for term in ["train", "valid", "test"]:
        with open("class/FB15k/freebase_mtr100_mte100-{}.txt".format(term), "r") as rf:
            for line in rf.readlines():
                line = line.strip().split("\t")
                KG.append(line)
    ent = [i[0] for i in KG] + [i[2] for i in KG]
    rel = [i[1] for i in KG]
    ent_order = reorder(ent)
    rel_order = reorder(rel)
    new_KG = [[ent_order[i[0]],rel_order[i[1]],ent_order[i[2]]] for i in KG]
    # data["A"] = new_KG

    ent_labels = []
    labels = []
    with open("class/FB15k/entity2type.txt", "r") as rf:
        for line in rf.readlines():
            line = line.strip().split("\t")
            ent_labels.append(line)
            labels += line[1:]
    labels = filter(labels)
    label_order = reorder(labels)
    new_ent_labels = []
    for each in ent_labels:
        each_label = []
        # print(each)
        for label in each[1:]:
            # print(label)
            if label in label_order:
                new_ent_labels.append([ent_order[each[0]], label_order[label]])
    data = np.array([1. for i in new_ent_labels])
    row = np.array([i[0] for i in new_ent_labels])
    col = np.array([i[1] for i in new_ent_labels])
    y = csr_matrix((data, (row, col)), shape=(len(ent_order), len(label_order)))
    # data["y"] = y

    train, test = [], []
    with open("class/FB15k/train.txt", "r") as rf:
        for line in rf.readlines():
            line = line.strip()
            train.append(ent_order[line])
    with open("class/FB15k/test.txt", "r") as rf:
        for line in rf.readlines():
            line = line.strip()
            test.append(ent_order[line])
    # data['train_idx'] = train
    # data['test_idx'] = test
    # data["e"] = len(ent_order)
    # print(train[:10])
    # print(test[:10])
    data = {'A': new_KG,
            'y': y,
            'train_idx': train,
            'test_idx': test,
            "e": len(ent_order)
            }
    with open('class/fb15kpro.pickle', 'wb') as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    dataset = sys.argv[1]
    align_relation(dataset)
    # process_fb()
    # process_wn()
