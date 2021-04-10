import math
from torch import nn
import torch
import numpy as np

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, edges):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edges = edges
        self.normalize_adjacency_matrix = self.norm_degs_matrix()
        self.fc = nn.Linear(in_features, out_features)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def norm_degs_matrix(self):
        I = np.identity(self.edges.shape[0]) #create Identity Matrix of A
        A = self.edges + I #add self-loop to A
        A = torch.Tensor(A) # convert np array to torch tensor

        node_degrees = A.sum(-1)
        degs_inv_sqrt = torch.pow(node_degrees, -0.5)
        norm_degs_matrix = torch.diag_embed(degs_inv_sqrt)
        normalize_adjacency_matrix = (norm_degs_matrix @ A @ norm_degs_matrix)
        return normalize_adjacency_matrix

    def forward(self, x):
        # support = torch.mm(x, self.weight)
        support = self.fc(x)
        device = support.get_device()
        normalize_adjacency_matrix = self.normalize_adjacency_matrix.to(f'cuda:{device}')
        output = torch.spmm(normalize_adjacency_matrix, support)
        return output


class GCN(nn.Module):
    def __init__(self, layer, layer_nums, edges):
        super(GCN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        gcs = []
        for i in range(layer):
            gcs.append(GraphConvolution(layer_nums[i], layer_nums[i+1], edges))
        self.gcs = torch.nn.ModuleList(gcs)

    def forward(self, x):
        for i, gc in enumerate(self.gcs):
            if i != len(self.gcs) - 1:
                x = self.relu(gc(x))
            else:
                x = gc(x)
        return x