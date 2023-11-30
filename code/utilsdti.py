import datetime
import dgl
import errno
import numpy as np
import pandas as pd
import os
import pickle
import random
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.metrics import auc as auc3
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio
from sklearn.metrics.pairwise import cosine_similarity as cos
import time
import scipy.spatial.distance as dist


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


default_configure = {
    'batch_size': 20
}


def setup(args, seed):
    args.update(default_configure)
    set_random_seed(seed)
    return args


def load_hetero(network_path):
    """
    meta_path of drug

    """
    drug_drug = np.loadtxt(network_path + 'mat_drug_drug.txt')
    drug_chemical = np.loadtxt(network_path + 'Similarity_Matrix_Drugs.txt') - 0.5
    drug_chemical = drug_chemical.clip(0, 1)
    drug_disease = np.loadtxt(network_path + 'mat_drug_disease.txt')
    disease_drug = drug_disease.T
    drug_sideeffect = np.loadtxt(network_path + 'mat_drug_se.txt')
    sideeffect_drug = drug_sideeffect.T
    drug_drug_protein = np.loadtxt(network_path + 'mat_drug_protein1.txt')

    """
    meta_path of protein

    """
    protein_protein = np.loadtxt(network_path + 'mat_protein_protein.txt')

    protein_sequence = np.loadtxt(network_path + 'Similarity_Matrix_Proteins.txt') / 100
    protein_sequence = (protein_sequence - 0.5) * 100
    protein_sequence = protein_sequence.clip(0, 1)

    protein_disease = np.loadtxt(network_path + 'mat_protein_disease.txt')
    disease_protein = protein_disease.T
    nums_di = len(disease_protein)
    d_d = dgl.graph(sparse.csr_matrix(drug_drug), ntype='drug', etype='similarity')
    num_drug = d_d.number_of_nodes()
    protein_protein_drug = drug_drug_protein.T
    d_c = dgl.graph(sparse.csr_matrix(drug_chemical), ntype='drug', etype='chemical')
    d_di = dgl.bipartite(sparse.csr_matrix(drug_disease), 'drug', 'ddi', 'disease')
    di_d = dgl.bipartite(sparse.csr_matrix(disease_drug), 'disease', 'did', 'drug')

    # d_d_p = dgl.bipartite(sparse.csr_matrix(drug_drug_protein), 'drug', 'dp', 'protein')

    d_se = dgl.bipartite(sparse.csr_matrix(drug_sideeffect), 'drug', 'dse', 'sideeffect')
    se_d = dgl.bipartite(sparse.csr_matrix(sideeffect_drug), 'sideeffect', 'sed', 'drug')

    p_p = dgl.graph(sparse.csr_matrix(protein_protein), ntype='protein', etype='similarity')
    num_protein = p_p.number_of_nodes()

    p_s = dgl.graph(sparse.csr_matrix(protein_sequence), ntype='protein', etype='sequence')
    p_di = dgl.bipartite(sparse.csr_matrix(protein_disease), 'protein', 'pdi', 'disease')
    p_d_d = dgl.bipartite(sparse.csr_matrix(protein_protein_drug), 'protein', 'pd', 'drug')
    d_d_p = dgl.bipartite(sparse.csr_matrix(drug_drug_protein), 'drug', 'dp', 'protein')

    di_p = dgl.bipartite(sparse.csr_matrix(disease_protein), 'disease', 'dip', 'protein')

    di_g = dgl.hetero_from_relations([di_d, d_di, di_p, p_di])

    # dg = dgl.hetero_from_relations([d_d, d_c, d_se, se_d, d_di, di_d,d_d_p,p_d_d])
    # pg = dgl.hetero_from_relations([p_p, p_s, p_di, di_p,p_d_d,d_d_p])
    dg = dgl.hetero_from_relations([d_d, d_c, d_se, se_d, d_di, di_d])
    pg = dgl.hetero_from_relations([p_p, p_s, p_di, di_p])
    graph = (dg, pg, di_g)

    # dti_o = torch.tensor(np.loadtxt(network_path + 'mat_drug_protein_train.txt'))

    # true_index = torch.where(dti_o == 1)
    # tep_label = torch.ones(len(true_index[0]), dtype=torch.long).reshape(-1, 1)
    # train_positive_index = torch.cat((true_index[0].type(dtype=torch.long).reshape(-1, 1),
    #                                   true_index[1].type(dtype=torch.long).reshape(-1, 1), tep_label), dim=1)
    # false_index = torch.where(dti_o == 0)
    # tep_label = torch.zeros(len(false_index[0]), dtype=torch.long).reshape(-1, 1)
    # whole_negative_index = torch.cat((false_index[0].reshape(-1, 1), false_index[1].reshape(-1, 1), tep_label), dim=1)
    #
    # negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
    #                                          size=len(train_positive_index),
    #                                          replace=False)
    #
    # data_set = np.concatenate((train_positive_index, whole_negative_index[negative_sample_index]), axis=0)
    # # print(len(negative_sample_index),len(data_set),data_set)
    # train = torch.tensor(data_set, dtype=torch.long)
    # f = open("/home/gyqiao/SGCL-DTI/data/heter/train.txt", "w")
    # for i in train.tolist():
    #     t = '	'.join(list(map(str, i)))
    #     f.write(f"{t}\n")
    # f.close()
    train = torch.tensor(np.loadtxt(network_path + 'train.txt'), dtype=torch.long)

    eval = torch.tensor(np.loadtxt(network_path + 'eval.txt'), dtype=torch.long)

    test = torch.tensor(np.loadtxt(network_path + 'test.txt'), dtype=torch.long)

    # test = torch.tensor(data_set, dtype=torch.long)

    node_num = [num_drug, num_protein, nums_di]

    all_meta_paths = [[["similarity"], ["chemical"], ['dse', 'sed'], ['ddi', 'did'], ],
                      [['similarity'], ["sequence"], ['pdi', 'dip']],
                      [['did', 'ddi'], ['dip', 'pdi']]
                      ]
    return (train, eval, test), graph, node_num, all_meta_paths, (drug_disease, protein_disease)


def load_homo(network_path, dataName):
    drug_protein = np.loadtxt(network_path + 'd_p.txt')
    protein_drug = drug_protein.T
    drug_drug = np.loadtxt(network_path + "d_d.txt")
    protein_protein = np.loadtxt(network_path + "p_p.txt")

    dti_o = torch.tensor(np.loadtxt(network_path + 'd_p_i.txt'))

    d_d = dgl.graph(sparse.csr_matrix(drug_drug), ntype='drug', etype='similarity')
    p_p = dgl.graph(sparse.csr_matrix(protein_protein), ntype='protein', etype='similarity')
    # d_p = dgl.bipartite(sparse.csr_matrix(drug_protein), 'drug', 'dp', 'protein')
    # p_d = dgl.bipartite(sparse.csr_matrix(protein_drug), 'protein', 'pd', 'drug')
    num_drug = d_d.number_of_nodes()
    num_protein = p_p.number_of_nodes()
    dg = dgl.hetero_from_relations([d_d])
    pg = dgl.hetero_from_relations([p_p])
    graph = [dg, pg]

    # true_index = torch.where(dti_o == 1)
    # tep_label = torch.ones(len(true_index[0]), dtype=torch.long).reshape(-1, 1)
    # train_positive_index = torch.cat((true_index[0].type(dtype=torch.long).reshape(-1, 1),
    #                                   true_index[1].type(dtype=torch.long).reshape(-1, 1), tep_label), dim=1)
    # false_index = torch.where(dti_o == 0)
    # tep_label = torch.zeros(len(false_index[0]), dtype=torch.long).reshape(-1, 1)
    # whole_negative_index = torch.cat((false_index[0].reshape(-1, 1), false_index[1].reshape(-1, 1), tep_label), dim=1)
    #
    # negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
    #                                          size=len(train_positive_index),
    #                                          replace=False)
    # lens = len(train_positive_index)
    # trlens = int(lens * 0.7)
    # telens = lens - trlens
    # train_ind = np.random.choice(np.arange(lens),size=trlens,replace=False,)
    # te_ind = [i for i in range(lens) if i not in train_ind]
    # te_ind = np.array(te_ind,dtype=int)
    #
    # data_set = np.concatenate((train_positive_index[train_ind], whole_negative_index[negative_sample_index][train_ind]),
    #                           axis=0)
    # # f = open("./train.txt", "w")
    # # for i in data_set.tolist():
    # #     t = '	'.join(list(map(str, i)))
    # #     f.write(f"{t}\n")
    # # f.close()
    # train = torch.tensor(data_set, dtype=torch.long)
    # #
    # data_set = np.concatenate((train_positive_index[te_ind], whole_negative_index[negative_sample_index][te_ind]),
    #                           axis=0)
    # f = open("./train.txt", "w")
    # for i in data_set.tolist():
    #     t = '	'.join(list(map(str, i)))
    #     f.write(f"{t}\n")
    # f.close()
    # test = torch.tensor(data_set, dtype=torch.long)
    train = torch.tensor(np.loadtxt(network_path + 'train.txt'), dtype=torch.long)

    eval = torch.tensor(np.loadtxt(network_path + 'eval.txt'), dtype=torch.long)

    test = torch.tensor(np.loadtxt(network_path + 'test.txt'), dtype=torch.long)

    node_num = [num_drug, num_protein]
    all_meta_paths = [[['similarity']],
                      [['similarity']]]
    return (train, eval, test), graph, node_num, all_meta_paths, None


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def get_roc(out, label):
    return roc_auc_score(label, out[:, 1:].detach().numpy())


def get_pr(out, label):
    precision, recall, thresholds = precision_recall_curve(label.cpu(), out[:, 1:].cpu().detach().numpy())
    return auc3(recall, precision)


def get_f1score(out, label):
    return f1_score(label.cpu(), out.argmax(dim=1).cpu().detach().numpy())


def get_L2reg(parameters):
    reg = 0
    for param in parameters:
        reg += 0.5 * (param ** 2).sum()
    return reg


def load_dataset(dateName):
    if dateName == "heter":
        return load_hetero("../data/heter/")
    else:
        return load_homo(f"../data/homo/{dateName}/", dateName)


def get_causal_index(tr, nums):
    tep_matrx = torch.zeros(nums, dtype=torch.int)
    for i in tr[tr[:, 2] == 1]:
        tep_matrx[i[0]][i[1]] = 1
    return (tep_matrx, tep_matrx.T)


def model_eval(model, d, p, node_feature, data, causal_index, graph):
    with torch.no_grad():
        index = data[-1]
        model.eval()
        test_label = torch.tensor(index[:, 2:3])
        out = model(graph, node_feature, index, causal_index, iftrain=False, d=d, p=p).cpu()

        acc = (out.argmax(dim=1) == test_label.reshape(-1)).sum(dtype=float) / len(test_label)
        roc = get_roc(out, test_label)
        pr = get_pr(out, test_label)
        return acc, roc, pr


def model_test(model, d, p, node_feature, data, epoch, causal_index, graph, bestacc):
    with torch.no_grad():
        index = data[-1]
        model.eval()
        test_label = torch.tensor(index[:, 2:3])
        out = model(graph, dataset_index=index, casual_index=causal_index, iftrain=False, d=d, p=p).cpu()

        acc = (out.argmax(dim=1) == test_label.reshape(-1)).sum(dtype=float) / len(test_label)
        roc = get_roc(out, test_label)
        pr = get_pr(out, test_label)
        return acc, roc, pr
