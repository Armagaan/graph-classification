import networkx as nx
import torch_geometric
import matplotlib.pyplot as plt
from torch_geometric.utils import from_networkx
from torch_geometric.data import Dataset
import torch
import numpy as np
import pickle
import os.path as osp

class isCyclicDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return 'dataset.pkl', 'labels.pkl'

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(951)] #951 graphs are present in the dataset

    def download(self):
        pass

    def process(self):
        print(self.raw_paths)
        self.data = pickle.load(open(self.raw_paths[0],'rb'))
        labels = pickle.load(open(self.raw_paths[1],'rb'))
        
        idx = 0
        for graph, label in zip(self.data, labels):
            # Read data from `raw_path`.
            nx.set_node_attributes(graph, 1.0, "x")
            data = from_networkx(graph)
            data.x = torch.reshape(data.x, (data.x.shape[0],1))
            data.y = torch.tensor(label)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.data)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

def generate_cyclic_graphs():
    '''cyclic graphs in networkx
        -complete_graph
        -circular_ladder_graph
        -circulant_graph
        -cycle_graph
        -lollipop_graph
        -wheel_graph
        -ladder_graph(n>2)
        -barbell_graph
        -grid_2d_graph
        -triangular_lattice_graph
    '''
    cyclic = []
    for i in range(3,53):
        cyclic.append(nx.cycle_graph(i))

    for i in range(3,53):
        cyclic.append(nx.complete_graph(i))

    for i in range(4,54):
        cyclic.append(nx.circular_ladder_graph(i))

    for i in range(2,10):
        for j in range(2,10):
            cyclic.append(nx.grid_2d_graph(i, j))

    for i in range(4,54):
        cyclic.append(nx.wheel_graph(i))

    for i in range(4,54):
        cyclic.append(nx.ladder_graph(i))

    for i in range(3,10):
        for j in range(3,10):
            cyclic.append(nx.lollipop_graph(i, j))

    for i in range(3,10):
        for j in range(3,10):
            cyclic.append(nx.triangular_lattice_graph(i, j, with_positions=False))

    for i in range(3,10):
        for j in range(3,10):
            cyclic.append(nx.barbell_graph(i, j))

    labels = np.ones(len(cyclic))
    return cyclic, list(labels)

def generate_acyclic_graphs():
    '''types of cyclic graphs
        -path_graph
        -star_graph
        -full_rary_tree
        -balanced_tree
        -binomial_tree
    '''
    acyclic = []

    for i in range(3,53):
        acyclic.append(nx.path_graph(i))

    for i in range(3,53):
        acyclic.append(nx.star_graph(i))

    for i in range(3, 10):
        for j in range(5, 55):
            acyclic.append(nx.full_rary_tree(i, j))

    for i in range(3, 10):
        for j in range(2, 7):
            acyclic.append(nx.full_rary_tree(i, j))

    for i in range(2, 7):
        acyclic.append(nx.binomial_tree(i))
    
    labels = np.zeros(len(acyclic))
    return acyclic, list(labels)

def create_isCyclic_dataset(saved=False):
    if(saved == False):
        cyclic, labels1 = generate_cyclic_graphs() 
        acyclic, labels0 = generate_acyclic_graphs()
        # nx.draw(cyclic[30])
        # plt.savefig('cyclic.png')
        # plt.clf()
        # nx.draw(acyclic[99])
        # plt.savefig('acyclic.png')

        dataset = cyclic + acyclic
        labels = labels1 + labels0
        f = open('data/isCyclic/raw/dataset.pkl', 'wb')
        lf = open('data/isCyclic/raw/labels.pkl', 'wb')
        pickle.dump(dataset,f)
        pickle.dump(labels,lf)
        f.close()
        lf.close()
        print("No of cyclic graphs: ", len(cyclic))
        print("No of acyclic graphs: ", len(acyclic))
        print("Dataset size: ", len(cyclic) + len(acyclic))
        # print(labels)
    else:
        f = open('data/isCyclic/raw/dataset.pkl', 'rb')
        lf = open('data/isCyclic/raw/labels.pkl', 'rb')
        dataset = pickle.load(f)
        labels = pickle.load(lf)
        f.close()
        lf.close()
        print("Dataset size: ", len(dataset))

    data_obj = isCyclicDataset(root ='data/isCyclic/')

    pytorch_data_obj = []
    for i in range(len(dataset)):
        # feats = np.ones(nx.number_of_nodes(dataset[i]))
        nx.set_node_attributes(dataset[i], 1.0, "x")
        # print(dataset[i].nodes().data())
        data = from_networkx(dataset[i])
        data.y = torch.tensor([labels[i]],dtype=torch.long)
        data.x = torch.reshape(data.x, (data.x.shape[0],1))
        # print(data)
        # print(data.edge_index, data.x, data.y)
        pytorch_data_obj.append(data)

    return data_obj, pytorch_data_obj
    # return dataset, pytorch_data_obj

dataset, data_objs = create_isCyclic_dataset(saved=True) #saved=True
print(dataset[0].x)
print(dataset[0].edge_index)
print(dataset[0].y)
df = open('data/isCyclic/isCyclic.pt', 'wb')
pickle.dump(data_objs, df)
df.close()

