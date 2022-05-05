import copy
import h5py
import numpy as np
np.seterr(invalid='ignore')

def load_data(data_path, normalize=False, scale_factors=None):
    x_dict_list, f_dict_list = [], []
    
    with h5py.File(data_path, 'r') as f:
        for idx in [idx for idx in f]:
            f_idx = f[idx]
            
            x_graph = {'globals': f_idx['x/globals'][()],
                         'nodes': f_idx['x/nodes'][()],
                         'edges': f_idx['x/edges'][()],
                       'senders': f_idx['x/senders'][()],
                     'receivers': f_idx['x/receivers'][()]}
            x_dict_list.append(x_graph)

            f_graph = {'globals': f_idx['f/globals'][()],
                         'nodes': f_idx['f/nodes'][()],
                         'edges': f_idx['f/edges'][()],
                       'senders': f_idx['f/senders'][()],
                     'receivers': f_idx['f/receivers'][()]}
            f_dict_list.append(f_graph)

    if normalize:
        x_dict_list, f_dict_list, _ = norm_data(xx=x_dict_list, ff=f_dict_list, scale_factors=scale_factors)
    
    return (x_dict_list, f_dict_list)

def norm_data(xx=None, ff=None, scale_factors=None):
    x, f = copy.deepcopy(xx), copy.deepcopy(ff)

    N_x = len(x) if (x is not None) else 0
    N_f = len(f) if (f is not None) else 0

    assert scale_factors is not None, 'Scale factors must be provided'

    for i in range(N_x):
        x[i]['edges'] = (x[i]['edges'] - scale_factors['x_edges'][:, 0])/scale_factors['x_edges'][:, 1]
        x[i]['nodes'] = (x[i]['nodes'] - scale_factors['x_nodes'][:, 0])/scale_factors['x_nodes'][:, 1]

        x[i]['globals'] = (x[i]['globals'] - scale_factors['x_globals'][:, 0])/scale_factors['x_globals'][:, 1]

    for i in range(N_f):
        f[i]['nodes'] = (f[i]['nodes'] - scale_factors['f_nodes'][:, 0])/scale_factors['f_nodes'][:, 1]
        f[i]['globals'] = (f[i]['globals'] - scale_factors['f_globals'][:, 0])/scale_factors['f_globals'][:, 1]

    if (N_x > 0) and (N_f > 0):
        return x, f, scale_factors
    elif (N_x > 0):
        return x, scale_factors
    elif (N_f > 0):
        return f, scale_factors

def unnorm_data(xx=None, ff=None, scale_factors=None):
    x, f = copy.deepcopy(xx), copy.deepcopy(ff)

    N_x = len(x) if (x is not None) else 0
    N_f = len(f) if (f is not None) else 0

    assert scale_factors is not None, 'Scale factors must be provided'

    for i in range(N_x):
        x[i]['edges'] = scale_factors['x_edges'][:, 1]*x[i]['edges'] + scale_factors['x_edges'][:, 0]
        x[i]['nodes'] = scale_factors['x_nodes'][:, 1]*x[i]['nodes'] + scale_factors['x_nodes'][:, 0]

        x[i]['globals'] = scale_factors['x_globals'][:, 1]*x[i]['globals'] + scale_factors['x_globals'][:, 0]

    for i in range(N_f):
        f[i]['nodes'] = scale_factors['f_nodes'][:, 1]*f[i]['nodes'] + scale_factors['f_nodes'][:, 0]
        f[i]['globals'] = scale_factors['f_globals'][:, 1]*f[i]['globals'] + scale_factors['f_globals'][:, 0]
 
    if (N_x > 0) and (N_f > 0):
        return x, f
    elif (N_x > 0):
        return x
    elif (N_f > 0):
        return f

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