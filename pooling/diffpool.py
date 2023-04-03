from math import ceil

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
from ogb.graphproppred.mol_encoder import AtomEncoder

from torch_geometric.nn import DenseGraphConv

from utils import fetch_assign_matrix, GCNConv


class SAGEConvolutions(nn.Module):  # define GraphSAGE là unit để dùng học biểu diễn và học clusters
    def __init__(self, num_layers,
                 in_channels,
                 out_channels,
                 residual=True):
        super().__init__()

        self.num_layers = num_layers
        self.residual = residual
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers - 1):  # number_layers > 1
            if i == 0:
                self.layers.append(DenseSAGEConv(in_channels, out_channels, normalize=True))
            else:
                self.layers.append(DenseSAGEConv(out_channels, out_channels, normalize=True))
            self.bns.append(nn.BatchNorm1d(out_channels))

        if num_layers == 1:  # when number_layers = 1
            self.layers.append(DenseSAGEConv(in_channels, out_channels, normalize=True))
        else:
            self.layers.append(DenseSAGEConv(out_channels, out_channels, normalize=True))

    def forward(self, x, adj, mask=None):  # x: embedding, adjacency matrix
        for i in range(self.num_layers - 1):  # forward qua n-1 layer
            x_new = F.relu(self.layers[i](x, adj, mask))
            batch_size, num_nodes, num_channels = x_new.size()
            x_new = x_new.view(-1, x_new.shape[-1])  # all data in the batch
            x_new = self.bns[i](x_new)
            x_new = x_new.view(batch_size, num_nodes, num_channels)
            if self.residual and x.shape == x_new.shape:  # add input into x_new
                x = x + x_new
            else:
                x = x_new
        x = self.layers[self.num_layers - 1](x, adj, mask)  # forward qua last layer
        return x


class DiffPoolLayer(nn.Module):  # su dụng SAGE convolution bên trên

    def __init__(self, dim_input, dim_embedding, no_new_clusters):
        super().__init__()

        # define two GNNs
        self.gnn_pool = SAGEConvolutions(1, dim_input, no_new_clusters)
        self.gnn_embed = SAGEConvolutions(1, dim_input, dim_embedding)

    def forward(self, x, adj, mask=None):
        s = self.gnn_pool(x, adj, mask)
        x = self.gnn_embed(x, adj, mask)
        x, adj, l, e = dense_diff_pool(x, adj, s, mask)
        return x, adj, l, e


class DiffPool(nn.Module):

    def __init__(self, num_features, num_classes, max_num_nodes, num_layers, gnn_hidden_dim,
                 gnn_output_dim, mlp_hidden_dim, pooling_type, encode_edge=False,
                 pre_sum_aggr=False):
        super().__init__()

        self.encode_edge = encode_edge
        self.max_num_nodes = max_num_nodes
        self.pooling_type = pooling_type
        self.num_pooling_layers = num_layers

        gnn_dim_input = num_features
        if encode_edge:
            gnn_dim_input = gnn_hidden_dim
            self.conv1 = GCNConv(gnn_hidden_dim, aggr='add')

        # Reproduce paper choice about coarse factor
        coarse_factor = 0.1 if num_layers == 1 else 0.25

        if pre_sum_aggr:  # this is only used for IMDB
            self.initial_embed = DenseGraphConv(gnn_dim_input, gnn_output_dim)
        else:
            self.initial_embed = SAGEConvolutions(1, gnn_dim_input, gnn_output_dim)

        no_new_clusters = ceil(coarse_factor * self.max_num_nodes)

        layers = []
        after_pool_layers = []

        for i in range(num_layers):  # 1 diffpool layers voi 3 SAGE convolution
            diffpool_layer = DiffPoolLayer(gnn_output_dim, gnn_output_dim, no_new_clusters)
            layers.append(diffpool_layer)

            # Update embedding sizes
            no_new_clusters = ceil(no_new_clusters * coarse_factor)

            after_pool_layers.append(SAGEConvolutions(3, gnn_output_dim, gnn_output_dim))

        self.diffpool_layers = nn.ModuleList(layers)
        self.after_pool_layers = nn.ModuleList(after_pool_layers)

        final_embed_dim_output = gnn_output_dim

        self.lin1 = nn.Linear(final_embed_dim_output, mlp_hidden_dim)
        self.lin2 = nn.Linear(mlp_hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.encode_edge:
            x = self.atom_encoder(x)
            x = self.conv1(x, edge_index, data.edge_attr)

        x, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index, batch=batch)

        x = self.initial_embed(x, adj, mask)

        x_all, l_total, e_total = [], 0, 0

        for i in range(self.num_pooling_layers):
            if i != 0:
                mask = None

            x, adj, l, e = self.diffpool_layers[i](x, adj, mask)  # x has shape (batch, MAX_no_nodes, feature_size)

            x = self.after_pool_layers[i](x, adj)

            l_total += l
            e_total += e

        x = torch.max(x, dim=1)[0]

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x, l_total, e_total
