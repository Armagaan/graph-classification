"""Read the Mutag dataset and create the graphx"""

import numpy as np
import os
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_networkx, to_dense_adj
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import random
import os.path as osp

# import dgl
# from dgl.data import DGLDataset
# import torch
# from dgl import save_graphs, load_graphs
# from utils.common_utils import read_file

def read_file(f_path):
    """
    read graph dataset .txt files
    :param f_path: the path to the .txt file
    :return: read the file (as lines) and return numpy arrays.
    """
    f_list = []
    with open(f_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.replace('\n', '').split(',')
            f_list.append([])
            for item in items:
                f_list[-1].append(int(item))
    return np.array(f_list).squeeze()

class NCI1Dataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return 'NCI1_A.txt', 'NCI1_graph_indicator.txt', 'NCI1_graph_labels.txt', 'NCI1_node_labels.txt'

    @property
    def processed_file_names(self):
        self.data_labels = read_file((self.raw_paths[2]))
        return [f'data_{i}.pt' for i in range(len(self.data_labels))]

    def download(self):
        pass

    def process(self):
        edge_data, graph_indicator, node_labels, graph_labels = self.nci1_preprocessing()
        self.edges = edge_data
        self.graph_indicator = graph_indicator
        self.node_labels = node_labels
        self.graph_labels = graph_labels
            
        self.graphs = [] #should contain x,edge_index of each graph
        self.labels = []
        self.feat_dim = len(np.unique(self.node_labels))
        self.graph_num = 0
        # group edges
        edges_group = {}
        for e_id, edge in enumerate(self.edges):
            g_id = self.graph_indicator[edge[0]]
            # print(g_id, e_id)
            if g_id != self.graph_indicator[edge[1]]:
                # print(self.graph_indicator[edge[1]])
                print('graph indicator error!', e_id, edge)
                exit(1)
            if g_id not in edges_group.keys():
                edges_group[g_id] = [edge]
            else:
                edges_group[g_id].append(edge)
        
        for g_id, g_edges in edges_group.items():
            g_label = self.graph_labels[g_id]
            g_edges = np.array(g_edges)
            src = g_edges[:, 0]
            dst = g_edges[:, 1]
            unique_nodes = np.unique(np.concatenate((src, dst), axis=0))
            g_feats = np.zeros((len(unique_nodes), self.feat_dim))
            int_feats = self.node_labels[unique_nodes]
            # print(unique_nodes, int_feats)
            
            g_feats[np.arange(len(unique_nodes)), int_feats] = 1
            #At this stage the indices in the edges variable are not normalized for the single graph, e.g. they do not start from 0
            edge_idx = torch.tensor(np.transpose(g_edges), dtype=torch.long)
            map_dict = {v.item():i for i,v in enumerate(unique_nodes)}
            map_edge = torch.zeros_like(edge_idx)
            for k,v in map_dict.items():
                map_edge[edge_idx==k] = v
            # print(map_dict)
            # continue
            edge_idx = map_edge.long()
            adj = to_dense_adj(edge_idx)
            # print(type(edge_idx), type(adj))
            # exit(0)
            g = Data(x=torch.tensor(g_feats, dtype=torch.float), edge_index=edge_idx, y=torch.tensor(g_label, dtype=torch.long)) #,adj=adj.squeeze(0), edge_weights = torch.from_numpy(edge_weights), edge_labels = torch.from_numpy(edge_labels))
            self.graphs.append(g)
            self.labels.append(g_label)
            torch.save(g, osp.join(self.processed_dir, f'data_{self.graph_num}.pt'))
            self.graph_num += 1

    def nci1_preprocessing(self):
        edge_path = self.raw_paths[0]
        graph_indicator_path = self.raw_paths[1]
        node_label_path = self.raw_paths[3]
        graph_label_path = self.raw_paths[2]
        edge_data = read_file(edge_path)
        edge_data = np.array(edge_data)
        edge_data = edge_data - 1
        graph_indicator = read_file(graph_indicator_path) - 1
        node_labels = np.array(read_file(node_label_path)) - 1
        graph_labels = read_file((graph_label_path))
        print(edge_path, graph_indicator_path, node_label_path, graph_label_path)
        # G_dataset = NCI1Dataset(edge_data, graph_indicator, node_labels, graph_labels)
        return edge_data, graph_indicator, node_labels, graph_labels
    
    def plot_graph(self, idx):
        data = self.get(idx)
        graph, label = data.x , data.y
        node_labels = torch.argmax(data.x, axis=1).detach().cpu().numpy()
        node_labels_dict = {}
        for i, lab in enumerate(node_labels):
            node_labels_dict[i] = {"label": int(lab)}
        g = to_networkx(data, to_undirected=True)
        nx.set_node_attributes(g, node_labels_dict)
        color_map = nx.get_node_attributes(g, "label")
        
        values = [color_map.get(node) for node in g.nodes()]

        pos = nx.spring_layout(g, seed=3113794652)
        plt.figure(figsize = (8,8))
        nx.draw(g, pos, cmap=plt.get_cmap('viridis'), node_color=values,node_size=80,linewidths=6)#with_labels=True
        plt.savefig(f'img_nci{idx}.png')

    def len(self):
        return len(self.data_labels)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


# # print('Loading ...')
# dataset = NCI1Dataset('data/NCI1/') # , 'data/NCI1/processed/'
# print('Number of graphs in dataset: ', len(dataset))
# dataset.plot_graph(3000)