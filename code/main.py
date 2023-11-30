# import dgl
from utilsdti import *
from modeltestdti import *
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from sklearn.metrics import roc_auc_score, f1_score
import warnings
import os
from sklearn.metrics.pairwise import cosine_similarity as cos

warnings.filterwarnings("ignore")
args = setup(default_configure, 53)
args['device'] = "cuda:1" if torch.cuda.is_available() else "cpu"
lamda = 0.01
in_size = 256
hidden_size = 128
out_size = 128
dropout = 0.5
lr = 0.0005
weight_decay = 0.000
epochs = 2000

fold = 0
dir = "../modelSave"
# ["heter","Es","GPCRs","ICs","Ns"]
for dataname in ["Es"]:
    if dataname == "heter":
        model_save_path = f"../new/{dataname}{seed}cls{lamda}/"
    else:
        model_save_path = f"../new/{dataname}{seed}cls{lamda}/"
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    dtidata, graph, num, all_meta_paths, causal_index = load_dataset(dataname)
    # dti_label = torch.tensor(dtidata[:, 2:3])
    # pd.to_pickle(dtidata, model_save_path + "data.pkl")
    hd = torch.randn((num[0], in_size))
    hp = torch.randn((num[1], in_size))
    features_d = hd.to(args['device'])
    features_p = hp.to(args['device'])

    if dataname == "heter":
        # for name in ["heter","Es","GPCRs","ICs","Ns"]:
        # dataName heter Es GPCRs ICs Ns zheng
        # print(all_meta_paths)
        hdi = torch.randn((num[2], in_size))
        features_di = hdi.to(args['device'])
        node_feature = [features_d, features_p, features_di]
        in_sizes = [hd.shape[1], hp.shape[1], hdi.shape[1]]
        hidden_sizes = [hidden_size, hidden_size, hidden_size]
        out_sizes = [out_size, out_size, out_size]
    else:
        in_sizes = [hd.shape[1], hp.shape[1]]
        hidden_sizes = [hidden_size, hidden_size]
        out_sizes = [out_size, out_size]
        node_feature = [features_d, features_p]


    def main(data, causal_index):
        i = 0
        file_save_path = f"{model_save_path}/fold{i}/"
        if dataname != "heter":
            causal_index = get_causal_index(data[0], num[:2])
        # else:
        # dp_metrx = get_causal_index(data[0], num[:2])
        # graph = construction_heter(relation, dp_metrx)
        if not os.path.exists(file_save_path):
            os.mkdir(file_save_path)

        pd.to_pickle(data, file_save_path + "index.pkl")
        model = HMTCL(
            all_meta_paths=all_meta_paths,
            in_size=in_sizes,
            hidden_size=hidden_sizes,
            out_size=out_sizes,
            dropout=dropout,
        ).to(args['device'])
        # model.load_state_dict(torch.load(f"{dir}/net{i}.pth"))
        optim = torch.optim.Adam(lr=lr, weight_decay=weight_decay, params=model.parameters())
        # losss = nn.BCELoss()
        best_acc = 0
        best_pr = 0
        best_roc = 0
        for epoch in tqdm(range(epochs)):
            model, loss, train_acc, task1_roc, acc, testroc, testpr = train(model, optim, data,node_feature, epoch, i,
                                                                            causal_index, graph,best_acc)

            best_acc = max(acc, best_acc)
            best_pr = max(testpr, best_pr)
            best_roc = max(testroc, best_roc)

        torch.save(model.state_dict(), f'{model_save_path}/fold{i}/net_params.pth')
        print(f"fold{i}  auroc is {best_roc.item():.4f} aupr is {best_pr.item():.4f} ")
        pd.to_pickle(f"fold{i}  auroc is {best_roc.item():.4f} aupr is {best_pr.item():.4f} ",
                     file_save_path + "rocvalue.pkl")


    # f.write(
    #     f"{name, s},{sum(all_acc) / len(all_acc):.4f},  {sum(all_roc) / len(all_roc):.4f} ,{sum(all_pr) / len(all_pr):.4f}")

    def train(model, optim, data, node_feature,epoch, fold, causal_index, graph,beseacc):
        model.train()
        train_index = data[0]
        out, d, p, cl_loss, a1, a2, a3 = model(graph, node_feature, train_index, causal_index)
        optim.zero_grad()
        train_label = torch.tensor(train_index[:, 2:3])
        loss = F.nll_loss(out, train_label.reshape(-1).to(args["device"])) + cl_loss * lamda
        loss.backward()
        optim.step()
        out = out.cpu()
        train_acc = (out.argmax(dim=1) == train_label.reshape(-1)).sum(dtype=float) / len(train_label)
        train_roc = get_roc(out, train_label)
        if epoch == epochs - 1:
            pd.to_pickle([d, p], f"{model_save_path}/fold{fold}/embedding.pkl")
            pd.to_pickle([a1, a2], f"{model_save_path}/fold{fold}/attention.pkl")

        eacc, eroc, epr = model_eval(model, d, p,node_feature, data[:2], epoch, causal_index, graph)
        racc, rroc, rpr = model_test(model, d, p,node_feature, data, epoch, causal_index, graph,beseacc)


        # print(
        #     f"{epoch} epoch train loss {loss.item():.3f}  train acc is {train_acc.item():.3f} test is acc  {racc.item():.4f}, test roc is {rroc.item():.4f}, test pr is {rpr.item():.4f}")

        return model, loss.item(), train_acc, train_roc, racc, rroc, rpr



    main(dtidata, causal_index )
