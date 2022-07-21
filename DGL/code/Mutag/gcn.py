import math

import dgl
from dgl.nn.pytorch import GraphConv
import numpy as np
import torch
from torch.nn import Linear
from torch.nn.functional import dropout, relu, softmax
from torch.nn.parameter import Parameter

# class GCNGraphNew(torch.nn.Module):
#     def __init__(self, in_feats, h_feats):
#         super(GCNGraphNew, self).__init__()
#         self.conv1 = GraphConv(in_feats, h_feats)
#         self.conv2 = GraphConv(h_feats, h_feats)
#         self.conv3 = GraphConv(h_feats, h_feats)
#         self.dense = torch.nn.Linear(h_feats, 1)
#         self.maxpool = dgl.nn.pytorch.glob.MaxPooling()

#     def forward(self, g, in_feat, e_weight):
#         h = self.conv1(g, in_feat, e_weight)
#         h = torch.nn.functional.relu(h)
#         h = self.conv2(g, h, e_weight)
#         h = torch.nn.functional.relu(h)
#         h = self.conv3(g, h, e_weight)
#         h = torch.nn.functional.relu(h)
#         g.ndata['h'] = h
#         h = self.maxpool(g, h)  # pooling
#         h = self.dense(h)
#         h = torch.nn.functional.sigmoid(h)
#         return h


class GraphConvLayer(torch.nn.Module):
    
    def __init__(self, shape_input: int, shape_output: int, bias: bool = True) -> None:
        super(GraphConvLayer, self).__init__()
        self.shape_input = shape_input
        self.shape_output = shape_output
        self.weight = Parameter(data=torch.Tensor(shape_input, shape_output))
        if bias:
            self.bias = Parameter(data=torch.Tensor(shape_output))
        else:
            self.register_parameter(name='bias', param=None)
        self.initialize_parameters()

    def initialize_parameters(self):
        std = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)
        return

    def forward(self, input_tensor: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        # * "adjacency_matrix" must be a sparse tensor with layout=torch.sparse_coo
        support = torch.mm(input_tensor, self.weight)
        output = torch.spmm(adjacency_matrix, support)
        if self.bias is not None:
            output += self.bias
        return output


class GCNGraph(torch.nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCNGraph, self).__init__()
        self.conv1 = GraphConvLayer(in_feats, h_feats)
        self.conv2 = GraphConvLayer(h_feats, h_feats)
        self.conv3 = GraphConvLayer(h_feats, h_feats)
        self.dense1 = torch.nn.Linear(h_feats, 16)
        self.dense2 = torch.nn.Linear(16, 8)
        self.dense3 = torch.nn.Linear(8, 1)

    def forward(self, g, in_feat, e_weight):
        mat_size = int(math.sqrt(e_weight.size(0)))
        dense = e_weight.reshape(mat_size, mat_size)
        sparse_adj = dense.to_sparse()

        h = self.conv1(in_feat, sparse_adj)
        h = torch.nn.functional.relu(h)
        h = self.conv2(h, sparse_adj)
        h = torch.nn.functional.relu(h)
        h = self.conv3(h, sparse_adj)
        g.ndata['h'] = h
        h = dgl.readout_nodes(g, 'h', op='mean') # pooling
        h = self.dense1(h)
        h = torch.nn.functional.relu(h)
        h = self.dense2(h)
        h = torch.nn.functional.relu(h)
        h = self.dense3(h)
        h = torch.sigmoid(h)
        return h


class GCNNodeBAShapes(torch.nn.Module):

    def __init__(self, in_feats, h_feats, out_feats, num_classes, device, if_exp=False):
        super(GCNNodeBAShapes, self).__init__()
        self.gc1 = GraphConvLayer(in_feats, h_feats)
        self.gc2 = GraphConvLayer(h_feats, h_feats)
        self.gc3 = GraphConvLayer(h_feats, out_feats)
        self.lin = Linear(
            in_features=h_feats + h_feats + out_feats,
            out_features=num_classes
        )
        self.if_exp = if_exp
        self.device = device

        self.training = True
        self.dropout = 0.5

    def forward(self, g, in_feat, e_weight, target_node):
        # map target node index
        x = torch.cat(
            (
                torch.tensor([0]).to(self.device),
                torch.cumsum(g.batch_num_nodes(), dim=0)
            ),
            dim=0
        )[:-1]
        target_node = target_node + x
        mat_size = int(math.sqrt(len(e_weight)))
        normalized_e_weight = self.normalize_adj(e_weight.reshape(mat_size, mat_size))

        h1 = relu(self.gc1(in_feat, normalized_e_weight))
        h1 = dropout(input=h1, p=self.dropout, training=self.training)
        h2 = relu(self.gc2(h1, normalized_e_weight))
        h2 = dropout(input=h2, p=self.dropout, training=self.training)
        h3 = self.gc3(h2, normalized_e_weight)
        h4 = self.lin(torch.cat((h1, h2, h3), dim=1))
        
        if self.if_exp:  # if in the explanation mod, should add softmax layer
            h4 = softmax(h4, dim=1)
        g.ndata['h'] = h4
        return g.ndata['h'][target_node]

    def get_degree_matrix(self, adj):
        return torch.diag(sum(adj))

    def normalize_adj(self, adj):
        # Normalize adjacancy matrix according to reparam trick in GCN paper
        A_tilde = adj + torch.eye(adj.shape[0])
        D_tilde = self.get_degree_matrix(A_tilde)
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
        return norm_adj


class GCNNodeTreeCycles(torch.nn.Module):

    def __init__(self, in_feats, h_feats, out_feats, num_classes, device, if_exp=False):
        super(GCNNodeTreeCycles, self).__init__()
        self.gc1 = GraphConvLayer(in_feats, h_feats)
        self.gc2 = GraphConvLayer(h_feats, h_feats)
        self.gc3 = GraphConvLayer(h_feats, out_feats)
        self.lin = Linear(
            in_features=h_feats + h_feats + out_feats,
            out_features=num_classes
        )
        self.if_exp = if_exp
        self.device = device

        self.training = True
        self.dropout = 0.5

    def forward(self, g, in_feat, e_weight, target_node):
        # map target node index
        x = torch.cat(
            (
                torch.tensor([0]).to(self.device),
                torch.cumsum(g.batch_num_nodes(), dim=0)
            ),
            dim=0
        )[:-1]
        target_node = target_node + x
        mat_size = int(math.sqrt(len(e_weight)))
        normalized_e_weight = self.normalize_adj(e_weight.reshape(mat_size, mat_size))

        h1 = relu(self.gc1(in_feat, normalized_e_weight))
        h1 = dropout(input=h1, p=self.dropout, training=self.training)
        h2 = relu(self.gc2(h1, normalized_e_weight))
        h2 = dropout(input=h2, p=self.dropout, training=self.training)
        h3 = self.gc3(h2, normalized_e_weight)
        h4 = self.lin(torch.cat((h1, h2, h3), dim=1))
        
        if self.if_exp:  # if in the explanation mod, should add softmax layer
            h4 = softmax(h4, dim=1)
        g.ndata['h'] = h4
        return g.ndata['h'][target_node]

    def get_degree_matrix(self, adj):
        return torch.diag(sum(adj))

    def normalize_adj(self, adj):
        # Normalize adjacancy matrix according to reparam trick in GCN paper
        A_tilde = adj + torch.eye(adj.shape[0])
        D_tilde = self.get_degree_matrix(A_tilde)
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
        return norm_adj


class GCNNodeTreeGrids(torch.nn.Module):

    def __init__(self, in_feats, h_feats, out_feats, num_classes, device, if_exp=False):
        super(GCNNodeTreeGrids, self).__init__()
        self.gc1 = GraphConvLayer(in_feats, h_feats)
        self.gc2 = GraphConvLayer(h_feats, h_feats)
        self.gc3 = GraphConvLayer(h_feats, out_feats)
        self.lin = Linear(
            in_features=h_feats + h_feats + out_feats,
            out_features=num_classes
        )
        self.if_exp = if_exp
        self.device = device

        self.training = True
        self.dropout = 0.5

    def forward(self, g, in_feat, e_weight, target_node):
        # map target node index
        x = torch.cat(
            (
                torch.tensor([0]).to(self.device),
                torch.cumsum(g.batch_num_nodes(), dim=0)
            ),
            dim=0
        )[:-1]
        target_node = target_node + x
        mat_size = int(math.sqrt(len(e_weight)))
        normalized_e_weight = self.normalize_adj(e_weight.reshape(mat_size, mat_size))

        h1 = relu(self.gc1(in_feat, normalized_e_weight))
        h1 = dropout(input=h1, p=self.dropout, training=self.training)
        h2 = relu(self.gc2(h1, normalized_e_weight))
        h2 = dropout(input=h2, p=self.dropout, training=self.training)
        h3 = self.gc3(h2, normalized_e_weight)
        h4 = self.lin(torch.cat((h1, h2, h3), dim=1))
        
        if self.if_exp:  # if in the explanation mod, should add softmax layer
            h4 = softmax(h4, dim=1)
        g.ndata['h'] = h4
        return g.ndata['h'][target_node]

    def get_degree_matrix(self, adj):
        return torch.diag(sum(adj))

    def normalize_adj(self, adj):
        # Normalize adjacancy matrix according to reparam trick in GCN paper
        A_tilde = adj + torch.eye(adj.shape[0])
        D_tilde = self.get_degree_matrix(A_tilde)
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
        return norm_adj


class GCNNodeCiteSeer(torch.nn.Module):
    # TODO
    def __init__(self, in_feats, h_feats, num_classes, if_exp=False):
        super(GCNNodeCiteSeer, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        self.if_exp = if_exp

    def forward(self, g, in_feat, e_weight, target_node):
        # map target node index
        x = torch.cat((torch.tensor([0]), torch.cumsum(g.batch_num_nodes(), dim=0)), dim=0)[:-1]
        target_node = target_node + x

        h = self.conv1(g, in_feat, e_weight)
        h = torch.nn.functional.relu(h)
        h = self.conv2(g, h, e_weight)
        if self.if_exp:  # if in the explanation mod, should add softmax layer
            h = torch.nn.functional.softmax(h)
        g.ndata['h'] = h
        return g.ndata['h'][target_node]
