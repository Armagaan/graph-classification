'''https://github.com/hdvvip/CS224W_Winter2021/blob/main/CS224W_Colab_2.ipynb'''
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool, GATConv, SAGEConv
import torch.nn.functional as F
from torch_geometric.utils import to_networkx 
from torch_geometric.datasets import TUDataset
from torch.nn import Linear, LogSoftmax 
from torch.nn import NLLLoss
# from ipywidgets import interact, interact_manual, FloatSlider
# from visualize_molecule import plot_mol
from dataset_class import isCyclicDataset, ADHDDataset, NCI1Dataset, isCyclicSmallDataset
from metrics import compute_accuracy
from operator import itemgetter
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import networkx as nx
import pandas as pd
import numpy as np
import pickle

torch.manual_seed(37)

class GNNGraph(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_dim,
        node_features_dim,
        edge_features_dim=None
    ):
        super(GNNGraph, self).__init__()
        self.hidden_dim = hidden_dim

        self.conv1 = GATConv(node_features_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        # self.conv3 = GCNConv(hidden_dim, hidden_dim)   
        # self.conv4 = GCNConv(hidden_dim, hidden_dim)
        # self.conv5 = GCNConv(hidden_dim, hidden_dim)

        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, num_classes)

        self.readout = LogSoftmax(dim=-1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # x = F.relu(self.conv3(x, edge_index))
        # x = F.relu(self.conv4(x, edge_index))
        # x = F.relu(self.conv5(x, edge_index))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        return self.readout(x)

class GNNGraphCustom(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        num_gnn_layers,
        hidden_dim,
        node_features_dim,
        edge_features_dim=None
    ):
        super(GNNGraphCustom, self).__init__()
        self.hidden_dim = hidden_dim

        self.gcn = nn.ModuleList()
        # added batching
        for i in range(num_gnn_layers):
            if (i == 0):
                self.gcn.append(GCNConv(node_features_dim, hidden_dim))
            else:
                self.gcn.append(GCNConv(hidden_dim, hidden_dim))

        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, num_classes)

        self.readout = LogSoftmax(dim=-1)

    def forward(self, x, edge_index, batch):
        for i, layer in enumerate(self.gcn):
            x = F.relu(layer(x, edge_index))  
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        return self.readout(x)
    
# dataset = isCyclicDataset(root ='data/isCyclic/') 
# for idx in range(len(dataset)):
#     if(dataset[idx].pos != None):
#         g = to_networkx(dataset[idx])
#         nx.draw(g)
#         plt.savefig(f'check_{idx}.png')
#         plt.clf()
# exit(0)

dataset_name = 'NCI1'

if(dataset_name == 'ADHD'):
    dataset = ADHDDataset(root ='data/ADHD/')
elif(dataset_name == 'isCyclic'): 
    dataset = isCyclicDataset(root ='data/isCyclic/') 
elif(dataset_name == 'NCI1'): 
   dataset =  NCI1Dataset(root ='data/NCI1/') 
elif(dataset_name == 'isCyclicSmall'): 
   dataset =  isCyclicSmallDataset(root ='data/isCyclicSmall/') 

print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs in dataset: {len(dataset)}')
print(f'Number of features: {dataset[0].x.shape[1]}')
# print(f'Number of features: {dataset.num_features}')
# print(f'Number of classes: {dataset.num_classes}')
print(f'Samples per class: {[ np.sum([1 for graph in dataset if graph.y == i ]) for i in range(2) ]}')

# If possible, we use a GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# for i in range(len(dataset)):
#     print(dataset[i])

train_size = int(len(dataset) * .5)
idx_train = torch.randint(low=0, high=len(dataset), size=[train_size])
idx_rest = []
idx_val = []
idx_test = []

for i in range(len(dataset)):
    if(i not in idx_train):
        idx_rest.append(i)

np.random.shuffle(idx_rest)
idx_val = torch.tensor(idx_rest[:int(len(idx_rest)*0.2)])
idx_test = torch.tensor(idx_rest[int(len(idx_rest)*0.2):])

print("Train idx: ", len(idx_train))
print("Val idx: ", len(idx_val))
print("Test idx: ", len(idx_test))

BATCH_SIZE = 128
BATCH_SIZE_TEST = idx_test.shape[0]
# print("BATCH SIZE TEST: ", BATCH_SIZE_TEST)

#itemgetter(*idx_val)(dataset)
# In the test loader we set the natch size to be equal to the size of the whole test set 
loader_train = DataLoader(dataset[idx_train], batch_size=BATCH_SIZE, shuffle=True)
loader_valid = DataLoader(dataset[idx_val], batch_size=BATCH_SIZE, shuffle=True)
loader_test = DataLoader(dataset[idx_test], batch_size=BATCH_SIZE_TEST, shuffle=False)

index = open(f'data/{dataset_name}/index.pkl', 'wb')
index_dict = {'idx_train':idx_train, 'idx_val':idx_val, 'idx_test':idx_test}
pickle.dump(index_dict, index)
index.close()

# Model
''' model = GNNGraph(
    num_classes = 3,
    hidden_dim=8,
    node_features_dim=1,
).to(device)'''

n_layers = 2

model = GNNGraphCustom(
    num_classes = 2,
    hidden_dim=16,
    num_gnn_layers=n_layers,
    node_features_dim=dataset[0].x.shape[1],
).to(device)

# Optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 50, 100, 150, 250, 400, 450], gamma=0.1, last_epoch=-1)

# Loss function
loss_function = NLLLoss()

def train_model(loader_train, loader_valid, model, optimizer, loss_function, N_EPOCHS):
  # Prepare empy lists for logging
    train_losses = []
    train_accs = []
    val_accs = []
    max_val_acc = 0

    for epoch in tqdm(range(N_EPOCHS)):
        epoch_loss = 0
        for batch in tqdm(loader_train, leave=False):
            batch.y = batch.y.type(torch.LongTensor) 
            batch.to(device)
            # print(batch.x.shape, batch.edge_index.shape, batch.y.shape)
            out = model(batch.x, batch.edge_index, batch.batch) #.float()
            loss = loss_function(out, batch.y.flatten())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_train = compute_accuracy(model, loader_train)
            acc_valid = compute_accuracy(model, loader_valid)
            if(acc_valid > max_val_acc):
                print("Updated: ", acc_valid)
                max_val_acc = acc_valid
                torch.save(model.state_dict(), f'models/gcn_{n_layers}layer_{dataset_name}.pt')

            with torch.no_grad():
                train_accs.append(acc_train)
                val_accs.append(acc_valid)
                train_losses.append(loss)

        print(f"Epoch: {epoch}, Loss: {loss}")

    # Visualization at the end of training
    fig, ax = plt.subplots(dpi=100)
    ax.plot(train_accs, c="steelblue", label="Training")
    ax.plot(val_accs, c="orangered", label="Validation")
    ax.grid()
    ax.legend()
    ax.set_title("Accuracy evolution")
    plt.savefig(f'GC_Accuracy_evolution_{dataset_name}.png')

N_EPOCHS = 500
train_model(loader_train, loader_valid, model, optimizer, loss_function, N_EPOCHS)
print("Test acc: ", compute_accuracy(model, loader_test))