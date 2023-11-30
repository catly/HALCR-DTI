# -*- coding: utf-8 -*-
from utilsdti import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv


device = "cuda:1" if torch.cuda.is_available() else "cpu"


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, z, ifaten=True):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)

        return (beta * z).flatten(1).squeeze()


class HANLayer(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        self.gat_layers1 = nn.ModuleList()
        #
        for i in range(len(meta_paths)):
            self.gat_layers.append(
                GATConv(in_size, 128, attn_drop=0.5, feat_drop=0.5, num_heads=1, activation=F.tanh))
        for i in range(len(meta_paths)):
            self.gat_layers1.append(
                GATConv(128, out_size, attn_drop=0.5, feat_drop=0.5, num_heads=1, activation=F.tanh))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                    g, meta_path)
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers1[i](new_g, self.gat_layers[i](new_g, h)))

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, dropout, num_heads=1):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads, dropout))
        self.predict = nn.Linear(hidden_size * num_heads, out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
        return self.predict(h.flatten(1))


class HAN_DTI(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout):
        super(HAN_DTI, self).__init__()
        self.sum_layers = nn.ModuleList()

        for i in range(0, len(all_meta_paths)):
            self.sum_layers.append(
                HAN(all_meta_paths[i], in_size[i], hidden_size[i], out_size[i], dropout))

    def forward(self, s_g, s_h):

        h1 = self.sum_layers[0](s_g[0], s_h[0])
        h2 = self.sum_layers[1](s_g[1], s_h[1])
        if len(s_h) == 3:
            h3 = self.sum_layers[2](s_g[2], s_h[2])
            return h1, h2, h3
        else:
            return h1, h2


class MLP(nn.Module):
    def __init__(self, nfeat):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(nfeat, nfeat // 2).apply(init),
            nn.ReLU(),
            nn.Linear(nfeat // 2, nfeat // 2).apply(init),
            nn.BatchNorm1d(nfeat//2)
        )

    def forward(self, x):
        output = F.normalize(self.MLP(x))
        return output


class MyClassifier(nn.Module):
    def __init__(self, nfeat):
        super(MyClassifier, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(nfeat, 128),
            nn.Tanh(),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=1),

        )

    def forward(self, x):
        output = self.MLP(x)
        return output



class HMTCL(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout):
        super(HMTCL, self).__init__()
        self.HAN_DTI = HAN_DTI(all_meta_paths, in_size, hidden_size, out_size, dropout)
        self.casual_project_drug = nn.Sequential(nn.Linear(out_size[0], out_size[0]), nn.Tanh()).apply(init)
        self.att = SemanticAttention(out_size[0])
        self.att1 = SemanticAttention(out_size[0] // 2)
        self.casual_project_pro = nn.Sequential(nn.Linear(out_size[0], out_size[0]), nn.Tanh()).apply(init)
        self.tau = 0.8
        self.proMLPe2 = MLP(out_size[0]*2)
        self.drugMlpe2 = MLP(out_size[0]*2)
        self.proMLPe1 = MLP(out_size[0])
        self.drugMlpe1 = MLP(out_size[0])
        self.classifier = MyClassifier(int(5* out_size[0]))

    def forward(self, graph, h=None, dataset_index=None, casual_index=None, iftrain=True, d=None, p=None):
        a1 = a2 = a3 = 0
        if iftrain:
            drug_di = torch.tensor(casual_index[0], dtype=torch.float).to(device)
            pro_di = torch.tensor(casual_index[1], dtype=torch.float).to(device)
            if len(h) == 3:
                d, p, di, a1, a2, a3 = self.HAN_DTI(graph, [h[0], h[1], h[2]])
                casual_drug = self.casual_project_drug((drug_di @ di) / (drug_di.sum(dim=1, keepdims=True) + 1e-8))
                casual_protein = self.casual_project_pro((pro_di @ di) / (pro_di.sum(dim=1, keepdims=True) + 1e-8))
            else:
                d, p = self.HAN_DTI(graph, [h[0], h[1]])
                casual_drug = self.casual_project_drug((drug_di @ p) / (drug_di.sum(dim=1, keepdims=True) + 1e-8))
                casual_protein = self.casual_project_pro((pro_di @ d) / (pro_di.sum(dim=1, keepdims=True) + 1e-8))

            de2 = self.att(torch.cat((d.unsqueeze(dim=1), casual_drug.unsqueeze(dim=1)), dim=1), False)
            pe2 = self.att(torch.cat((p.unsqueeze(dim=1), casual_protein.unsqueeze(dim=1)), dim=1), False)
            de2 = self.drugMlpe2(de2)
            pe2 = self.proMLPe2(pe2)
            de1 = self.drugMlpe1(casual_drug)
            pe1 = self.proMLPe1(casual_protein)
            loss = self.cl_loss(d, de2) + self.cl_loss(p, pe2)

            d = torch.cat((de1, de2, d), dim=1)
            p = torch.cat((pe1, pe2, p), dim=1)

        drug_index = dataset_index[:, 0]
        pro_index = dataset_index[:, 1]
        # pred = torch.sigmoid(d[drug_index] @ p[pro_index].T).diag().reshape(-1,1).cpu()
        # pred = (d[drug_index] @ p[pro_index].T).diag().reshape(-1,1).cpu()
        pred = self.classifier(torch.cat((d[drug_index], p[pro_index]), dim=1))
        if iftrain:
            return pred, d, p, loss, a1, a2, a3
        return pred

    def cl_loss(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        sim_matrix = sim_matrix / (torch.sum(sim_matrix, dim=1).view(-1, 1) + 1e-6)
        sim_matrix = sim_matrix.to(device)

        loss = -torch.log(sim_matrix.mul(torch.eye(dot_numerator.shape[0]).to(device)).sum(dim=-1)).mean()
        return loss


def init(i):
    if isinstance(i, nn.Linear):
        torch.nn.init.xavier_uniform_(i.weight)
