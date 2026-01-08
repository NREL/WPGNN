import os
import h5py
import numpy as np
from time import time
import modules as mod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LeakyReLU

import torch_geometric.nn as pyg_nn
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer

from torch_geometric.data import Data
from torch_geometric.profile import get_model_size, count_parameters


class WPGNN(nn.Module):
    '''
        Parameters:
            eN_in, eN_out   - number of input/output edge features
            nN_in, nN_out   - number of input/output node features
            gN_in, gN_out   - number of input/output graph features
            n_layers        - number of graph layers in the network
            graph_layers    - list of graph layers
            model_path      - location of a saved model, if None then use randomly initialized weights
            scale_factors   - list of scaling factors used to normalize data
            optmizer        - Sonnet optimizer object that will be used for training
    '''
    def __init__(self, eN=2, nN=3, gN=3, graph_size=None,
                       scale_factors=None, model_path=None, name=None):
        super(WPGNN, self).__init__()

        # Set model architecture
        self.eN_in,  self.nN_in,  self.gN_in  = eN, nN, gN
        if graph_size is None:
            graph_size = [[32, 32, 32],
                          [16, 16, 16],
                          [16, 16, 16],
                          [ 8,  8,  8],
                          [ 8,  8,  8],
                          [ 4,  2,  2]]
        self.n_layers = len(graph_size)
        self.eN_out, self.nN_out, self.gN_out = graph_size[-1][0], graph_size[-1][1], graph_size[-1][2]

        # Construct WPGNN model
        # Edge-node-global update with MetaLayer instead of graph_layers
        self.meta_layers = nn.ModuleList()
        for i in range(self.n_layers - 1):
            dim_in = [self.eN_in, self.nN_in, self.gN_in] if i == 0 else graph_size[i-1]
            dim_out = graph_size[i]
            self.meta_layers.append(MetaLayer(
                                        edge_model=mod.EdgeModel(dim_in, dim_out, n_layers=2, output_activation='sigmoid', layer_index=i),
                                        node_model=mod.NodeModel(dim_in, dim_out, n_layers=2, output_activation='sigmoid', layer_index=i),
                                        global_model=mod.GlobalModel(dim_in, dim_out, n_layers=2, output_activation='sigmoid', layer_index=i),
                                    ))
        
        self.meta_layers.append(MetaLayer(
                                    edge_model=mod.EdgeModel(graph_size[-2], graph_size[-1], n_layers=1, output_activation='relu', layer_index=self.n_layers-1),
                                    node_model=mod.NodeModel(graph_size[-2], graph_size[-1], n_layers=1, output_activation='relu', layer_index=self.n_layers-1),
                                    global_model=mod.GlobalModel(graph_size[-2], graph_size[-1], n_layers=1, output_activation='relu', layer_index=self.n_layers-1),
                                ))
        
        

    def forward(self, data):
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.y, data.batch
        
        # Reshape tensor (B * F_u) to (B, F_u)
        u = u.reshape(-1, 3)

        for meta_layer in self.meta_layers:
            x, edge_attr, u = meta_layer(x, edge_index, edge_attr, u, batch)
        
        # return output graph
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=u.reshape(-1), batch=batch)
    
    def loss(self, pred, label):
        # Compute the mean squared error for the target turbine- and plant-level outputs
        # Need to reshape y to (B, 2) for both pred and label to compute loss on each feature
        pred.y = pred.y.reshape(-1, 2)
        label.y = label.y.reshape(-1, 2)
        
        turbine_loss = torch.mean((pred.x - label.x) ** 2, dim=0)
        plant_loss = torch.mean((pred.y - label.y) ** 2, dim=0)

        loss = torch.sum(plant_loss) + 10.*torch.sum(turbine_loss)

        # Note that x has [Power, Wind Speed], and y has [Total Power, MST distance]
        return loss, turbine_loss[0], turbine_loss[1], plant_loss[0], plant_loss[1]

    def model_summary(self):
        print(f"Model size: {get_model_size(self)} MB")
        print(f"Number of parameters: {count_parameters(self)}")

    
    def fit(self, train_loader, test_loader, optimizer, scheduler=None, n_epochs=10, save_model_path=None, batch_reporting=False):
        '''
            Parameters:
                train_data       - training data in (list of input graphs, list of output graphs) format
                test_data        - test data used to monitor training progress, same format as training data
                batch_size       - number of samples to include in each training batch
                learning_rate    - learning rate for the training optimizer
                decay_rate       - rate of decay for the learning rate
                epochs           - the total number of epochs of training to perform
                print_every      - how frequently (in training iterations) to print the batch performance
                save_every       - how frequently (in epochs) to save the model
                save_model_path  - path to directory where to save model during training
        '''
        # Start training process
        self.train()
        for epoch in range(n_epochs):
            start_time = time()
            total_loss, total_loss_tp, total_loss_ts, total_loss_pp, total_loss_pc = 0.0, 0.0, 0.0, 0.0, 0.0
            print(f"\nEpoch {epoch+1}\n" + "-"*60)

            for idx_batch, batch in enumerate(train_loader):
                x_batch, f_batch = batch    # Dataset consists of (graph_x, graph_f) pairs
                optimizer.zero_grad()

                # Model returns (nodes, edges, globals)
                pred = self(x_batch)
                loss = self.loss(pred, f_batch)

                if batch_reporting:
                    print(f"Batch {idx_batch+1}:")
                    print('Total batch loss = {:.6f}'.format(loss[0]))
                    print('Turbine power loss = {:.6f}, '.format(loss[1]), 'turbine speed loss = {:.6f}'.format(loss[2]))
                    print('Plant power loss   = {:.6f}, '.format(loss[3]), 'plant cabling loss = {:.6f}'.format(loss[4]))
            
                loss[0].backward()
                optimizer.step()    # Update weights
                total_loss += loss[0].item() * f_batch.num_graphs       # f_batch.num_graphs = len(f_batch) = batch_size
                total_loss_tp += loss[1].item() * f_batch.num_graphs
                total_loss_ts += loss[2].item() * f_batch.num_graphs
                total_loss_pp += loss[3].item() * f_batch.num_graphs
                total_loss_pc += loss[4].item() * f_batch.num_graphs
                
                # Save current state of the model

            scheduler.step() if scheduler is not None else None     # Update LR after the epoch finishes
            if save_model_path is not None:
                torch.save(self.state_dict(), f"{save_model_path}/wpgnn_epoch{epoch+1:03d}.pt")
            
            # Report epoch losses
            total_loss /= len(train_loader.dataset)
            total_loss_tp /= len(train_loader.dataset)
            total_loss_ts /= len(train_loader.dataset)
            total_loss_pp /= len(train_loader.dataset)
            total_loss_pc /= len(train_loader.dataset)
            if isinstance(loss, tuple):
                print(f"Training Loss: {total_loss:.6f}")
                print(f"Turbine Power Loss: {total_loss_tp:.6f} | Turbine Speed Loss: {total_loss_ts:.6f} | Plant Power Loss: {total_loss_pp:.6f} | Plant Cabling Loss: {total_loss_pc:.6f}")
            else:
                print(f"Training Loss: {total_loss:.6f}")

        print("\n" + "="*60)
        print("Training Complete!")
        print('Time to complete: {0:02f}\n'.format(time() - start_time), flush=True)
        print("="*60)
    