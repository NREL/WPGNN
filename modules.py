import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LeakyReLU
from torch_geometric.utils import scatter


class EdgeModel(nn.Module):
    def __init__(self, dim_in, dim_out, n_layers=3, output_activation='relu', layer_index=0):
        super().__init__()
        # Input: edge + 2*node + global
        edge_inputs, edge_outputs = dim_in[0] + 2*dim_in[1] + dim_in[2], dim_out[0]
        layer_sizes = [edge_outputs] if n_layers == 1 else [edge_outputs for _ in range(n_layers-1)]
        self.n_layers = len(layer_sizes)
        self.output_activation = output_activation

        self.edge_mlp = []
        for i in range(self.n_layers):
            self.edge_mlp.append(Lin(edge_inputs if i == 0 else layer_sizes[i-1], layer_sizes[i]))
            self.edge_mlp.append(LeakyReLU())
        self.edge_mlp.append(Lin(layer_sizes[-1], edge_outputs))
        if self.output_activation == 'leaky_relu':
            self.edge_mlp.append(LeakyReLU())
        elif self.output_activation == 'relu':
            self.edge_mlp.append(ReLU())
        elif self.output_activation == 'softplus':
            self.edge_mlp.append(nn.Softplus())
        elif self.output_activation == 'sigmoid':
            self.edge_mlp.append(nn.Sigmoid())
        self.edge_mlp = Seq(*self.edge_mlp)
    
    def forward(self, src, dst, edge_attr, u, batch):
        # src, dst: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [E] with max entry B - 1.
        
        # u = u.unsqueeze(0).repeat(src.size(0), 1) if u.dim() == 1 else u        # Make u the same size as src/dst
        # print("Shape check on EdgeModel: ", src.shape, dst.shape, edge_attr.shape, u.shape)
        out = torch.cat([src, dst, edge_attr, u[batch]], dim=1)
        out = self.edge_mlp(out)

        return out

class NodeModel(nn.Module):
    def __init__(self, dim_in, dim_out, n_layers=3, output_activation='relu', layer_index=0):
        super().__init__()

        # Input: 2*edge + node + global
        # Note: Using dim_out[0] as edges have already been updated
        node_inputs, node_outputs = 2*dim_out[0] + dim_in[1] + dim_in[2], dim_out[1]
        layer_sizes = [node_outputs] if n_layers == 1 else [node_outputs for _ in range(n_layers-1)]
        self.n_layers = len(layer_sizes)
        self.output_activation = output_activation

        self.node_mlp = []
        for i in range(self.n_layers):
            self.node_mlp.append(Lin(node_inputs if i == 0 else layer_sizes[i-1], layer_sizes[i]))
            self.node_mlp.append(LeakyReLU())
        self.node_mlp.append(Lin(layer_sizes[-1], node_outputs))
        if self.output_activation == 'leaky_relu':
            self.node_mlp.append(LeakyReLU())
        elif self.output_activation == 'relu':
            self.node_mlp.append(ReLU())
        elif self.output_activation == 'softplus':
            self.node_mlp.append(nn.Softplus())
        elif self.output_activation == 'sigmoid':
            self.node_mlp.append(nn.Sigmoid())
        self.node_mlp = Seq(*self.node_mlp)
    
    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [N] with max entry B - 1.
        row, col = edge_index

        # Aggregate received edges (incoming messages to each node)
        received = scatter(edge_attr, col, dim=0, dim_size=x.size(0), reduce='mean')

        # Aggregate sent edges (outgoing messages to each node)
        sent = scatter(edge_attr, row, dim=0, dim_size=x.size(0), reduce='mean')

        # Concatenate: received +sent + node + global[batch]
        # print("Shape check on NodeModel: ", received.shape, sent.shape, x.shape, u.shape)
        out = torch.cat([received, sent, x, u[batch]], dim=1)
        out = self.node_mlp(out)

        return out


class GlobalModel(nn.Module):
    def __init__(self, dim_in, dim_out, n_layers=3, output_activation='relu', layer_index=0):
        super().__init__()
        
        # Input: edge + node + global
        # Note: Using dim_out[0] and dim_out[1] as edges and nodes have already been updated
        global_inputs, global_outputs = dim_out[0] + dim_out[1] + dim_in[2], dim_out[2]
        layer_sizes = [global_outputs] if n_layers == 1 else [global_outputs for _ in range(n_layers-1)]
        self.n_layers = len(layer_sizes)
        self.output_activation = output_activation

        self.global_mlp = []
        for i in range(self.n_layers):
            self.global_mlp.append(Lin(global_inputs if i == 0 else layer_sizes[i-1], layer_sizes[i]))
            self.global_mlp.append(LeakyReLU())
        self.global_mlp.append(Lin(layer_sizes[-1], global_outputs))
        if self.output_activation == 'leaky_relu':
            self.global_mlp.append(LeakyReLU())
        elif self.output_activation == 'relu':
            self.global_mlp.append(ReLU())
        elif self.output_activation == 'softplus':
            self.global_mlp.append(nn.Softplus())
        elif self.output_activation == 'sigmoid':
            self.global_mlp.append(nn.Sigmoid())
        self.global_mlp = Seq(*self.global_mlp)

    
    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [N] with max entry B - 1.
        
        row, col = edge_index
        n_aggr = scatter(x, batch, dim=0, dim_size=u.size(0), reduce='mean')
        e_aggr = scatter(edge_attr, batch[col], dim=0, dim_size=u.size(0), reduce='mean')
        out = torch.cat([e_aggr, n_aggr, u], dim=1)
        out = self.global_mlp(out)

        return out