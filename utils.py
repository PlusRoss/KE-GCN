import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import tensorflow as tf
import math
import os
import random
from collections import Counter

flags = tf.app.flags
FLAGS = flags.FLAGS


def create_exp_dir(path, scripts_to_save=None):
    path_split = path.split("/")
    path_i = "."
    for one_path in path_split:
        path_i += "/" + one_path
        if not os.path.exists(path_i):
            os.mkdir(path_i)

    print('Experiment dir : {}'.format(path_i))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def inverse_sum(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    return d_inv_sqrt.reshape((-1, 1))


def preprocess_adj(adj):
    ent_adj_invsum = inverse_sum(adj[0])
    rel_adj_invsum = inverse_sum(adj[1])
    return [ent_adj_invsum, rel_adj_invsum, adj[2]]


def construct_feed_dict(features, support, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    if isinstance(support[0], list):
        for i in range(len(support)):
            feed_dict.update({placeholders['support'][i][j]: support[i][j] \
                                    for j in range(len(support[i]))})
    else:
        feed_dict.update({placeholders['support'][i]: support[i] \
                                for i in range(len(support))})
    return feed_dict


def loadfile(file, num=1):
    '''
    num: number of elements per row
    '''
    print('loading file ' + file)
    ret = []
    with open(file, "r", encoding='utf-8') as rf:
        for line in rf:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def get_ent2id(files):
    ent2id = {}
    for file in files:
        with open(file, 'r', encoding='utf-8') as rf:
            for line in rf:
                th = line[:-1].split('\t')
                ent2id[th[1]] = int(th[0])
    return ent2id


def get_extended_adj_auto(e, KG):
    nei_list = []
    ent_row, rel_row = [], []
    ent_col, rel_col = [], []
    ent_data, rel_data = [], []
    count = 0
    for tri in KG:
        nei_list.append([tri[0], tri[1], tri[2]])
        ent_row.append(tri[0])
        ent_col.append(count)
        ent_data.append(1.)
        ent_row.append(tri[2])
        ent_col.append(count)
        ent_data.append(1.)
        rel_row.append(tri[1])
        rel_col.append(count)
        rel_data.append(1.)
        count += 1
    ent_adj_ind = sp.coo_matrix((ent_data, (ent_row, ent_col)), shape=(e, count))
    rel_adj_ind = sp.coo_matrix((rel_data, (rel_row, rel_col)), shape=(max(rel_row)+1, count))
    return [ent_adj_ind, rel_adj_ind, np.array(nei_list)]


def load_data_class(FLAGS):

    def analysis(A, y, train, test):
        for A_i in A:
            print(A_i.nonzero())
        exit()

    def to_KG(A):
        KG = []
        count = 0
        for A_i in A:
            idx = A_i.nonzero()
            for head, tail in zip(idx[0], idx[1]):
                KG.append([head, count, tail])
            if len(idx[0]) > 0:
                count += 1
        # print(KG[:100])
        return KG

    dirname = os.path.dirname(os.path.realpath(sys.argv[0]))
    raw_file = dirname + '/data/class/' + FLAGS.dataset + '.pickle'
    pro_file = dirname + '/data/class/' + FLAGS.dataset + 'pro.pickle'

    if not os.path.exists(pro_file):
        with open(dirname + '/data/class/' + FLAGS.dataset + '.pickle', 'rb') as f:
            data = pkl.load(f)
        A = data['A']
        KG = to_KG(A)
        num_ent = A[0].shape[0]
        data["A"] = KG
        data["e"] = num_ent
        # analysis(A, y, train, test)
        with open(dirname + '/data/class/' + FLAGS.dataset + 'pro.pickle', 'wb') as handle:
            pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)

    with open(dirname + '/data/class/' + FLAGS.dataset + 'pro.pickle', 'rb') as f:
        data = pkl.load(f)

    KG = data["A"]
    # y: csr_sparse_matrix
    y = sp.csr_matrix(data['y']).astype(np.float32)
    train = data['train_idx']
    test = data['test_idx']
    label_ind = train + test
    num_ent = data["e"]
    # print("train class: ", Counter(np.argmax(np.asarray(y[train].todense()), 1).reshape(1,-1)[0]))
    if FLAGS.dataset in ["wordnet", "fb15k"]:
        random.shuffle(label_ind)
        split = [0.1, 0.2]
        train = label_ind[:int(split[0]*len(label_ind))]
        valid = label_ind[int(split[0]*len(label_ind)):int(split[1]*len(label_ind))]
        test = label_ind[int(split[1]*len(label_ind)):]
        print("train {}, valid {}, test {}".format(len(train), len(valid), len(test)))
    else:
        valid = None

    adj = get_extended_adj_auto(num_ent, KG)

    return adj, num_ent, train, test, valid, y


def load_data_align(FLAGS):
    names = [['ent_ids_1', 'ent_ids_2'], ['triples_1', 'triples_2'], ['ref_ent_ids']]
    if FLAGS.rel_align:
        names[1][1] = "triples_2_relaligned"
    for fns in names:
        for i in range(len(fns)):
            fns[i] = 'data/'+FLAGS.dataset+'/'+fns[i]
    Ent_files, Tri_files, align_file = names
    num_ent = len(set(loadfile(Ent_files[0], 1)) | set(loadfile(Ent_files[1], 1)))
    align_labels = loadfile(align_file[0], 2)
    num_align_labels = len(align_labels)
    np.random.shuffle(align_labels)
    if not FLAGS.valid:
        train = np.array(align_labels[:num_align_labels // 10 * FLAGS.seed])
        valid = None
    else:
        train = np.array(align_labels[:int(num_align_labels // 10 * (FLAGS.seed-1))])
        valid = align_labels[int(num_align_labels // 10 * (FLAGS.seed-1)): num_align_labels // 10 * FLAGS.seed]
    test = align_labels[num_align_labels // 10 * FLAGS.seed:]
    KG = loadfile(Tri_files[0], 3) + loadfile(Tri_files[1], 3)
    ent2id = get_ent2id([Ent_files[0], Ent_files[1]])
    adj = get_extended_adj_auto(num_ent, KG)
    return adj, num_ent, train, test, valid


def load_data_rel_align(FLAGS):
    names = [['ent_ids_1', 'ent_ids_2'], ['triples_1', 'triples_2'], ['ref_ent_ids']]
    for fns in names:
        for i in range(len(fns)):
            fns[i] = 'data/'+FLAGS.dataset+'/'+fns[i]
    Ent_files, Tri_files, align_file = names
    num_ent = len(set(loadfile(Ent_files[0], 1)) | set(loadfile(Ent_files[1], 1)))
    align_labels = loadfile(align_file[0], 2)
    num_align_labels = len(align_labels)
    np.random.shuffle(align_labels)
    if not FLAGS.valid:
        train = np.array(align_labels[:num_align_labels // 10 * FLAGS.seed])
        valid = None
    else:
        train = np.array(align_labels[:int(num_align_labels // 10 * (FLAGS.seed-1))])
        valid = align_labels[int(num_align_labels // 10 * (FLAGS.seed-1)): num_align_labels // 10 * FLAGS.seed]
    test = align_labels[num_align_labels // 10 * FLAGS.seed:]
    KG = loadfile(Tri_files[0], 3) + loadfile(Tri_files[1], 3)
    ent2id = get_ent2id([Ent_files[0], Ent_files[1]])
    adj = get_extended_adj_auto(num_ent, KG)
    rel_align_labels = loadfile('data/'+FLAGS.dataset+"/ref_rel_ids", 2)
    num_rel_align_labels = len(rel_align_labels)
    np.random.shuffle(rel_align_labels)
    if not FLAGS.valid:
        train_rel = np.array(rel_align_labels[:num_rel_align_labels // 10 * FLAGS.rel_seed])
        valid_rel = None
    else:
        train_rel = np.array(rel_align_labels[:int(num_rel_align_labels // 10 * (FLAGS.rel_seed-1))])
        valid_rel = rel_align_labels[int(num_rel_align_labels // 10 * (FLAGS.rel_seed-1)): num_rel_align_labels // 10 * FLAGS.rel_seed]
    test_rel = rel_align_labels[num_rel_align_labels // 10 * FLAGS.rel_seed:]
    return adj, num_ent, train, test, valid, train_rel, test_rel, valid_rel
