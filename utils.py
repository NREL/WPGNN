import copy
import h5py
import pickle
import torch
import numpy as np
from glob import glob
from time import time
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch_geometric.utils

np.seterr(invalid='ignore')

def load_data(filename, normalize=True, sf=None):
    '''
    Load wind plant data from HDF5 or pickle file and return as list of graph pairs (g_x, g_f).
    Each graph pair corresponds to an input-output pair for the WPGNN model.

    Input graph g_x:
        Node features: (N_turbs, 3) -> [X location, Y location, Yaw angle]
        Edge features: (N_edges, 2) -> [dx, dy] relative positions
        Graph features: (3,) -> [u, v, TI]
        ========================================================
        Shape: Data(x=[N_turbs, 3], edge_index=[2, N_edges], edge_attr=[N_edges, 2], y=[3])
    
    Output graph g_f:
        Node features: (N_turbs, 2) -> [Power, Wind Speed]
        Edge features: (N_edges, 2) -> [Generalized features]
        Graph features: (2,) -> [Total Power, MST distance]
        ========================================================
        Shape: Data(x=[N_turbs, 2], edge_index=[2, N_edges], edge_attr=[N_edges, 2], y=[2])
    '''
    file_type = filename.split('.')[-1]
    if file_type == 'pkl':
        with open(filename, 'rb') as f:
            data = pickle.load(f)

            if normalize:
                data_norm = []
                for g_x, g_f in data:
                    g_x, g_f = norm_data(g_x=g_x, g_f=g_f, sf_x=sf['x'], sf_f=sf['f'])
                    data_norm.append((g_x, g_f))
                data = data_norm

        return data

    data = []
    with h5py.File(filename, 'r') as hf:
        layouts = [layout for layout in hf if 'Layout' in layout]
        layouts.sort()
        layouts = layouts

        for layout in layouts:
            hf_layout = hf[layout]
            N_turbs = hf_layout['Number of Turbines'][()]

            X_x_baseline = np.zeros((N_turbs, 3))
            turbines = [s for s in hf_layout['Turbines']]
            turbines.sort()
            for turbine in turbines:
                turb_idx = int(turbine[-3:])
                turb_x = hf_layout['Turbines'][turbine]['X Location'][()]
                turb_y = hf_layout['Turbines'][turbine]['Y Location'][()]

                X_x_baseline[turb_idx, 0] = turb_x
                X_x_baseline[turb_idx, 1] = turb_y

            D = distance_matrix(X_x_baseline[:, :2], X_x_baseline[:, :2])
            mst_dist = np.sum(minimum_spanning_tree(D))

            scenarios = [s for s in hf_layout['Scenarios']]
            scenarios.sort()
            scenarios = scenarios
            for scenario in scenarios:

                ws = hf_layout['Scenarios'][scenario]['Wind Speed'][()]
                wd = hf_layout['Scenarios'][scenario]['Wind Direction'][()]
                ti = hf_layout['Scenarios'][scenario]['Turbulence Intensity'][()]
                uv = speed_to_velocity(np.array([ws, wd]))
                U_x = np.array(list(uv)+[ti])

                turb_yaw = hf_layout['Scenarios'][scenario]['Yaw Angles'][()]
                X_x = np.copy(X_x_baseline)
                X_x[:, 2] = turb_yaw

                turb_pow = hf_layout['Scenarios'][scenario]['Turbine Power'][()]
                turb_ws = hf_layout['Scenarios'][scenario]['Turbine Wind Speed'][()]
                X_f = np.stack([turb_pow, turb_ws]).T

                U_f = np.array([np.sum(turb_pow), mst_dist])

                A_x, senders, receivers = identify_edges(X_x[:, :2], wd, cone_deg=15)
                edge_index = np.stack([senders, receivers])

                A_f = np.zeros((A_x.shape[0], 2))
                
                g_x = Data(y=torch.tensor(U_x, dtype=torch.float),
                           x=torch.tensor(X_x, dtype=torch.float), 
                           edge_index=torch.tensor(edge_index, dtype=torch.int),
                           edge_attr=torch.tensor(A_x, dtype=torch.float))
                g_f = Data(y=torch.tensor(U_f, dtype=torch.float),
                           x=torch.tensor(X_f, dtype=torch.float), 
                           edge_index=torch.tensor(edge_index, dtype=torch.int),
                           edge_attr=torch.tensor(A_f, dtype=torch.float))

                if normalize:
                    g_x, g_f = norm_data(g_x=g_x, g_f=g_f, sf_x=sf['x'], sf_f=sf['f'])

                data.append((g_x, g_f))

    #with open('data/small_data.pkl', 'wb') as f:
    #    pickle.dump(data, f)

    return data

def speed_to_velocity(xx):
    x = np.atleast_2d(copy.deepcopy(xx))

    ws, wd = x[:, 0], -(x[:, 1]+90)*(np.pi/180.)
    u, v = -ws*np.cos(wd), -ws*np.sin(wd)

    if x.shape[0] == 1:
        x = np.concatenate((u, v), axis=0)
    else:
        x = np.concatenate((np.atleast_2d(u), np.atleast_2d(v)), axis=0).T

    return x

def velocity_to_speed(xx):
    x = np.atleast_2d(copy.deepcopy(xx))

    u, v = x[:, 0], x[:, 1]
    ws = np.sqrt(u**2 + v**2)

    wd = 90-np.arctan(v/u)*(180./np.pi)
    wd[u<0] += 180

    # If ws = 0, then no way to recover the direction
    wd[np.isnan(wd)] = 0.

    if x.shape[0] == 1:
        x = np.concatenate((ws, wd), axis=0)
    else:
        x = np.concatenate((np.atleast_2d(ws), np.atleast_2d(wd)), axis=0).T

    return x

def identify_edges(x_loc, wind_dir, cone_deg=15):
    # Identify edges where wake interactions may play a role in power generation
    N_turbs = x_loc.shape[0]

    u, v = speed_to_velocity([10., wind_dir])
    theta = np.arctan(v/u)
    if u < 0:
        theta += np.pi
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    x_loc = x_loc@R

    x_rel = x_loc.reshape((1, N_turbs, 2)) - x_loc.reshape((N_turbs, 1, 2))

    alpha = np.arctan(x_rel[:, :, 1]/x_rel[:, :, 0])*(180./np.pi)
    alpha[np.isnan(alpha)] = 90.

    directed_edge_indices = ((abs(alpha) < cone_deg) & (x_rel[:, :, 0] <= 0)).nonzero()

    senders, receivers = directed_edge_indices[0], directed_edge_indices[1]

    edges = x_rel[senders, receivers, :]

    return edges, senders, receivers

def norm_data(g_x=None, g_f=None, sf_x=None, sf_f=None):
    assert (g_x is not None) or (g_f is not None)

    if g_x is not None:
        assert sf_x is not None
    
        U_x, X_x, A_x = g_x.y.numpy(), g_x.x.numpy(), g_x.edge_attr.numpy()
        edge_index = g_x.edge_index.numpy()

        U_x = (U_x - sf_x['graph'][0])/sf_x['graph'][1]
        X_x = (X_x - sf_x['node'][0])/sf_x['node'][1]
        A_x = (A_x - sf_x['edge'][0])/sf_x['edge'][1]

        g_x = Data(y=torch.tensor(U_x, dtype=torch.float),
                   x=torch.tensor(X_x, dtype=torch.float), 
                   edge_index=torch.tensor(edge_index, dtype=torch.int),
                   edge_attr=torch.tensor(A_x, dtype=torch.float))

    if g_f is not None:
        assert sf_f is not None
    
        U_f, X_f, A_f = g_f.y.numpy(), g_f.x.numpy(), g_f.edge_attr.numpy()
        edge_index = g_f.edge_index.numpy()

        U_f = (U_f - sf_f['graph'][0])/sf_f['graph'][1]
        X_f = (X_f - sf_f['node'][0])/sf_f['node'][1]
        A_f = (A_f - sf_f['edge'][0])/sf_f['edge'][1]

        g_f = Data(y=torch.tensor(U_f, dtype=torch.float),
                   x=torch.tensor(X_f, dtype=torch.float), 
                   edge_index=torch.tensor(edge_index, dtype=torch.int),
                   edge_attr=torch.tensor(A_f, dtype=torch.float))

    if (g_x is not None) and (g_f is not None):
        return g_x, g_f
    elif (g_x is not None):
        return g_x
    elif (g_x is not None):
        return g_f

def unnorm_data(g_x=None, g_f=None, sf_x=None, sf_f=None):
    assert (g_x is not None) or (g_f is not None)

    if g_x is not None:
        assert sf_x is not None
    
        U_x, X_x, A_x = g_x.y.numpy(), g_x.x.numpy(), g_x.edge_attr.numpy()
        edge_index = g_x.edge_index.numpy()

        U_x = sf_x['graph'][1]*U_x + sf_x['graph'][0]
        X_x = sf_x['node'][1]*X_x + sf_x['node'][0]
        A_x = sf_x['edge'][1]*A_x + sf_x['edge'][0]

        g_x = Data(y=torch.tensor(U_x, dtype=torch.float),
                   x=torch.tensor(X_x, dtype=torch.float), 
                   edge_index=torch.tensor(edge_index, dtype=torch.int),
                   edge_attr=torch.tensor(A_x, dtype=torch.float))

    if g_f is not None:
        assert sf_f is not None
    
        U_f, X_f, A_f = g_f.y.numpy(), g_f.x.numpy(), g_f.edge_attr.numpy()
        edge_index = g_f.edge_index.numpy()

        U_f = sf_f['graph'][1]*U_f + sf_f['graph'][0]
        X_f = sf_f['node'][1]*X_f + sf_f['node'][0]
        A_f = sf_f['edge'][1]*A_f + sf_f['edge'][0]

        g_f = Data(y=torch.tensor(U_f, dtype=torch.float),
                   x=torch.tensor(X_f, dtype=torch.float), 
                   edge_index=torch.tensor(edge_index, dtype=torch.int),
                   edge_attr=torch.tensor(A_f, dtype=torch.float))

    if (g_x is not None) and (g_f is not None):
        return g_x, g_f
    elif (g_x is not None):
        return g_x
    elif (g_x is not None):
        return g_f



def visualize_graphs(graphs):
    for idx, (g_x, g_f) in enumerate(graphs):
        # Two subplots to visualize a pair of graphs: g_x and g_f
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        # Plot g_x
        G_x = torch_geometric.utils.to_networkx(g_x)
        nx.draw(G_x, ax=axes[0], with_labels=True, node_color='lightblue', edge_color='gray')
        # x = g_x.x.numpy()
        # edge_index = g_x.edge_index.numpy()

        # axes[0].scatter(x[:, 0], x[:, 1], s=100, c='blue')
        # for i in range(edge_index.shape[1]):
        #     src = edge_index[0, i]
        #     dst = edge_index[1, i]
        #     axes[0].plot([x[src, 0], x[dst, 0]], [x[src, 1], x[dst, 1]], c='gray', linestyle='--')
        # axes[0].set_title('G_x {}'.format(idx+1))
        # axes[0].set_xlabel('X Location (m)')
        # axes[0].set_ylabel('Y Location (m)')
        # axes[0].axis('equal')
        # axes[0].grid(True)

        # Plot g_f
        G_f = torch_geometric.utils.to_networkx(g_f)
        nx.draw(G_f, ax=axes[1], with_labels=True, node_color='lightgreen', edge_color='gray')
        # x = g_f.x.numpy()
        # edge_index = g_f.edge_index.numpy()

        # axes[1].scatter(x[:, 0], x[:, 1], s=100, c='red')
        # for i in range(edge_index.shape[1]):
        #     src = edge_index[0, i]
        #     dst = edge_index[1, i]
        #     axes[1].plot([x[src, 0], x[dst, 0]], [x[src, 1], x[dst, 1]], c='gray', linestyle='--')
        # axes[1].set_title('G_f {}'.format(idx+1))
        # axes[1].set_xlabel('X Location (m)')
        # axes[1].set_ylabel('Y Location (m)')
        # axes[1].axis('equal')
        # axes[1].grid(True)

        plt.savefig('data/graphs/graph_{}.png'.format(idx+1))    # Make directory 'graphs' if not exists
        plt.close()

        break
    