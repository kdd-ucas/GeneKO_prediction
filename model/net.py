
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv,GATConv

import  torch.nn.functional as F
device = torch.device("cuda:3")

class NET(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels,n_feature, out):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels,heads=8, concat=False)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=8, concat=False)

        self.encoder = nn.Sequential(
            nn.Linear(n_feature, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512,256),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out),
            nn.ReLU(),
        )

    def decode_re(self,x):
        x = self.encoder(x)
        y = self.decoder(x)
        return x, y


    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, training=self.training)
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
        # return torch.sigmoid(torch.mm(z[edge_label_index[0]],z[edge_label_index[1]]))

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def enco(self, a, b):
        # adj = a @ (b.t())
        adj = torch.mm(a,(b.t()))
        # print(adj)
        return adj


