import os
import h5py
import numpy as np
from time import time
import tensorflow as tf
import sonnet as snt
from graph_nets import blocks
from graph_nets.utils_tf import *
from graph_nets.utils_np import graphs_tuple_to_data_dicts
import modules as mod
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float32')

class WPGNN(snt.Module):
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
        super(WPGNN, self).__init__(name=name)

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
        self.graph_layers = []
        for i in range(self.n_layers - 1):
            dim_in = [self.eN_in, self.nN_in, self.gN_in] if i == 0 else graph_size[i-1]
            self.graph_layers.append(self.graph_layer(dim_in, graph_size[i],
                                                      n_layers=2,
                                                      output_activation='sigmoid',
                                                      layer_index=i))
        self.graph_layers.append(self.graph_layer(graph_size[-2], graph_size[-1],
                                                  n_layers=1,
                                                  output_activation='relu',
                                                  layer_index=i+1))

        # Pass data through model to inialize network weights
        tmp_input = [{'globals': np.random.normal(size=(self.gN_in)),
                        'nodes': np.random.normal(size=(2, self.nN_in)),
                        'edges': np.random.normal(size=(2, self.eN_in)),
                      'senders': [1, 0],
                    'receivers': [0, 1]}]
        self(data_dicts_to_graphs_tuple(tmp_input))

        # Load save model weights
        if model_path is not None:
            self.custom_load_weights(model_path)

        if scale_factors is None:
            self.scale_factors = {'x_globals': np.array([[0., 25.], [0., 25.], [0.09, 0.03]]),
                                    'x_nodes': np.array([[0., 75000.], [0., 85000.], [15., 15.]]),
                                    'x_edges': np.array([[-100000., 100000.], [0., 75000.]]),
                                  'f_globals': np.array([[0., 500000000.], [0., 100000.]]),
                                    'f_nodes': np.array([[0., 5000000.], [0.,25.]]),
                                    'f_edges': np.array([[0., 0.]])}
        else:
            self.scale_factors = scale_factors

        self.optimizer = snt.optimizers.Adam()

    def __call__(self, graph_in, physical_units=False):
        # Evaluate the WPGNN on a given input graph
        for i_layer, graph_layer in enumerate(self.graph_layers):
            phi_e = graph_layer[0]
            phi_n = graph_layer[1]
            phi_g = graph_layer[2]

            graph_out = phi_g(phi_n(phi_e(graph_in)))
            
            tf_edge_dims = (graph_in.edges.shape[1] == graph_out.edges.shape[1])
            tf_node_dims = (graph_in.nodes.shape[1] == graph_out.nodes.shape[1])
            tf_global_dims = (graph_in.globals.shape[1] == graph_out.globals.shape[1])
            if tf_edge_dims & tf_node_dims & tf_global_dims :
                graph_out = graph_out.replace(edges=graph_in.edges+graph_out.edges)
                graph_out = graph_out.replace(nodes=graph_in.nodes+graph_out.nodes)
                graph_out = graph_out.replace(globals=graph_in.globals+graph_out.globals)

            graph_in = graph_out
        
        return graph_in

    def graph_layer(self, dim_in, dim_out, n_layers=3, output_activation='relu', layer_index=0):
        edge_inputs, edge_outputs = dim_in[0] + 2*dim_in[1] + dim_in[2], dim_out[0]
        layer_sizes = [edge_outputs for _ in range(n_layers-1)]
        edge_update = mod.EdgeUpdate(-1, edge_outputs, layer_sizes=layer_sizes,
                                     output_activation=output_activation,
                                     name='edgeUpdate{0:02d}'.format(layer_index))
        phi_e = blocks.EdgeBlock(edge_model_fn=lambda: edge_update,
                                 use_edges=True, 
                                 use_receiver_nodes=True, 
                                 use_sender_nodes=True, 
                                 use_globals=True)
        
        node_inputs, node_outputs = 2*dim_in[0] + dim_in[1] + dim_in[2], dim_out[1]
        layer_sizes = [node_outputs for _ in range(n_layers-1)]
        node_update = mod.NodeUpdate(-1, node_outputs, layer_sizes=layer_sizes,
                                     output_activation=output_activation,
                                     name='nodeUpdate{0:02d}'.format(layer_index))
        phi_n = blocks.NodeBlock(node_model_fn=lambda: node_update,
                                 use_received_edges=True, 
                                 use_sent_edges=True, 
                                 use_nodes=True, 
                                 use_globals=True)

        global_inputs, global_outputs = dim_in[0] + dim_in[1] + dim_in[2], dim_out[2]
        layer_sizes = [global_outputs for _ in range(n_layers-1)]
        global_update = mod.GlobalUpdate(-1, global_outputs, layer_sizes=layer_sizes,
                                         output_activation=output_activation,
                                         name='globalUpdate{0:02d}'.format(layer_index))
        phi_g = blocks.GlobalBlock(global_model_fn=lambda: global_update,
                                   use_edges=True, 
                                   use_nodes=True, 
                                   use_globals=True)
        
        phi_g(phi_n(phi_e(data_dicts_to_graphs_tuple([{'globals': np.random.normal(size=(dim_in[2])),
                                                         'nodes': np.random.normal(size=(2, dim_in[1])),
                                                         'edges': np.random.normal(size=(2, dim_in[0])),
                                                       'senders': [1, 0],
                                                     'receivers': [0, 1]}]))))
        
        return [phi_e, phi_n, phi_g]
    
    def custom_save_weights(self, filename):
        all_weights = [weight for graph_layer in self.graph_layers \
                              for block in graph_layer \
                              for weight in block.trainable_variables]
        
        with h5py.File(filename, 'w') as f:
            for weight in all_weights:
                w_name = weight.name.split('/')
                w_path, w_name = '/'.join(w_name[:-1]), w_name[-1]
                try:
                    f.create_group(w_path)
                except:
                    pass
                f[w_path].create_dataset(w_name, data=weight.numpy())

    def custom_load_weights(self, filename):
        all_weights = [weight for graph_layer in self.graph_layers \
                              for block in graph_layer \
                              for weight in block.trainable_variables]

        with h5py.File(filename, 'r') as f:
            for weight in all_weights:
                weight.assign(f[weight.name][()])
    
    def compute_dataset_loss(self, data, batch_size=100, reporting=False):
        # Compute the mean loss across an entire data without updating the model parameters
        ds = tf.data.Dataset.from_tensor_slices(np.arange(len(data[0]))).batch(batch_size)

        if reporting:
            N, l_tot, l_tp_tot, l_ts_tot, l_pp_tot, l_ps_tot = 0., 0., 0., 0., 0., 0.
        else:
            N, l_tot = 0., 0.

        for idx_batch in ds:
            x_batch = data_dicts_to_graphs_tuple([data[0][idx] for idx in idx_batch])
            f_batch = data_dicts_to_graphs_tuple([data[1][idx] for idx in idx_batch])

            N_batch = len(idx_batch)

            l = self.compute_loss(x_batch, f_batch, reporting=reporting)

            if reporting:
                l_tot += l[0]*N_batch
                l_tp_tot += l[1]*N_batch
                l_ts_tot += l[2]*N_batch
                l_pp_tot += l[3]*N_batch
                l_ps_tot += l[4]*N_batch
            else:
                l_tot += l*N_batch
            N += N_batch

        if reporting:
            return l_tot/N, l_tp_tot/N, l_ts_tot/N, l_pp_tot/N, l_ps_tot/N
        else:
            return l_tot/N
    
    def compute_loss(self, x, f, reporting=False):
        # Compute the mean squared error for the target turbine- and plant-level outputs
        x_out = self(x)
        
        turbine_loss = tf.reduce_mean((x_out.nodes - f.nodes)**2, axis=0)
        plant_loss = tf.reduce_mean((x_out.globals - f.globals)**2, axis=0)
        
        loss = tf.reduce_sum(plant_loss) + 10.*tf.reduce_sum(turbine_loss)

        if reporting:
            return loss, turbine_loss[0], turbine_loss[1], plant_loss[0], plant_loss[1]
        else:
            return loss

    def train_step(self, x, f):
        with tf.GradientTape() as tape:
            l = self.compute_loss(x, f)
            grad = tape.gradient(l, self.trainable_variables)
        self.optimizer.apply(grad, self.trainable_variables)
        
    def fit(self, train_data, test_data=None, batch_size=100, learning_rate=1e-3, decay_rate=0.99,
                  epochs=100, print_every=10, save_every=100, save_model_path=None):
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
        self.optimizer.learning_rate = learning_rate

        # Build data pipelines
        train_ds = tf.data.Dataset.from_tensor_slices(np.arange(len(train_data[0]))).shuffle(10000).batch(batch_size)

        # Start training process
        iters = 0
        for epoch in range(1, epochs+1):
            start_time = time()
            print('Beginning epoch {}...'.format(epoch))

            for idx_batch in train_ds:
                x_batch = data_dicts_to_graphs_tuple([train_data[0][idx] for idx in idx_batch])
                f_batch = data_dicts_to_graphs_tuple([train_data[1][idx] for idx in idx_batch])

                self.train_step(x_batch, f_batch)

                if (print_every > 0) and ((iters % print_every) == 0):
                    l = self.compute_loss(x_batch, f_batch, reporting=True)
                    print('Total batch loss = {:.6f}'.format(l[0]))
                    print('Turbine power loss = {:.6f}, '.format(l[1]), 'turbine speed loss = {:.6f}'.format(l[2]))
                    print('Plant power loss   = {:.6f}, '.format(l[3]), 'plant cabling loss = {:.6f}'.format(l[4]))
                    print('')

                iters += 1
            
            # Save current state of the model
            if (save_model_path is not None) and ((epoch % save_every) == 0):
                model_epoch = save_model_path+'/{0:05d}'.format(epoch)
                if not os.path.exists(model_epoch):
                    os.makedirs(model_epoch)
                self.custom_save_weights('/'.join([model_epoch, 'wpgnn.h5']))
            
            # Report current training/testing performance of model
            l = self.compute_dataset_loss(train_data, batch_size=batch_size, reporting=True)
            print('Epochs {} Complete'.format(epoch))
            print('Training Loss = {:.6f}, '.format(l[0]))
            print('Turbine power loss = {:.6f}, '.format(l[1]), 'turbine speed loss = {:.6f}'.format(l[2]))
            print('Plant power loss   = {:.6f}, '.format(l[3]), 'plant cabling loss = {:.6f}'.format(l[4]))
            
            if test_data is not None:
                l = self.compute_dataset_loss(test_data, batch_size=batch_size, reporting=True)
                print('Testing Loss = {:.6f}, '.format(l[0]))
                print('Turbine power loss = {:.6f}, '.format(l[1]), 'turbine speed loss = {:.6f}'.format(l[2]))
                print('Plant power loss   = {:.6f}, '.format(l[3]), 'plant cabling loss = {:.6f}'.format(l[4]))
            
            self.optimizer.learning_rate *= decay_rate

            print('Time to complete: {0:02f}\n'.format(time() - start_time), flush=True)


