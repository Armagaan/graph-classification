"""
1. Custom GCN layer.
2. GNN for NCI1 dataset.
"""

import math
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch_geometric.nn import global_mean_pool, GCNConv

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_Mutag(nn.Module):
    def __init__(self, in_features, h_features, n_classes) -> None:
        super(GCN_Mutag, self).__init__()
        self.conv1 = GraphConvolution(in_features, h_features)
        self.conv2 = GraphConvolution(h_features, h_features)
        self.conv3 = GraphConvolution(h_features, h_features)
        self.dense1 = Linear(h_features, 16)
        self.dense2 = Linear(16, 8)
        self.dense3 = Linear(8, 1)

    def forward(self, feature_matrix, edge_index, batch):
        dense_adj = torch.sparse.FloatTensor(edge_index, torch.ones(edge_index.size(1)))
        x = self.conv1(feature_matrix, dense_adj)
        x = x.relu()
        x = self.conv2(x, dense_adj)
        x = x.relu()
        x = self.conv3(x, dense_adj)
        x = global_mean_pool(x, batch=batch)
        x = self.dense1(x)
        x = x.relu()
        x = self.dense2(x)
        x = x.relu()
        x = self.dense3(x)
        x = torch.sigmoid(x)
        return x
