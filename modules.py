import tensorflow as tf
import sonnet as snt

class EdgeUpdate(snt.Module):

    def __init__(self, input_size, output_size, layer_sizes=None, output_activation=False, 
                       w_init=None, b_init=None, name=None):
        super(EdgeUpdate, self).__init__(name=name)
        self.input_size, self.output_size = input_size, output_size
        self.output_activation = output_activation

        layer_sizes = [output_size] if layer_sizes is None else layer_sizes
        self.n_layers = len(layer_sizes)

        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(snt.Linear(layer_sizes[i], 
                                          w_init=w_init,
                                          b_init=b_init, 
                                          name='linear{0:03d}'.format(i)))
        self.layers.append(snt.Linear(output_size, 
                                      w_init=w_init,
                                      b_init=b_init, 
                                      name='linear_out'))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = tf.nn.leaky_relu(x)

        x = self.layers[-1](x)
        if self.output_activation == 'leaky_relu':
            x = tf.nn.leaky_relu(x)
        elif self.output_activation == 'relu':
            x = tf.nn.relu(x)
        elif self.output_activation == 'softplus':
            x = tf.nn.softplus(x)
        elif self.output_activation == 'sigmoid':
            x = tf.nn.sigmoid(x)
        elif self.output_activation == 'none':
            pass
        else:
            assert self.output_activation == False

        return x

class NodeUpdate(snt.Module):

    def __init__(self, input_size, output_size, layer_sizes=None, output_activation=False, 
                       w_init=None, b_init=None, name=None):
        super(NodeUpdate, self).__init__(name=name)
        self.input_size, self.output_size = input_size, output_size
        self.output_activation = output_activation

        layer_sizes = [output_size] if layer_sizes is None else layer_sizes
        self.n_layers = len(layer_sizes)

        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(snt.Linear(layer_sizes[i], 
                                          w_init=w_init,
                                          b_init=b_init, 
                                          name='linear{0:03d}'.format(i)))
        self.layers.append(snt.Linear(output_size, 
                                      w_init=w_init,
                                      b_init=b_init, 
                                      name='linear_out'))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = tf.nn.leaky_relu(x)

        x = self.layers[-1](x)
        if self.output_activation == 'leaky_relu':
            x = tf.nn.leaky_relu(x)
        elif self.output_activation == 'relu':
            x = tf.nn.relu(x)
        elif self.output_activation == 'softplus':
            x = tf.nn.softplus(x)
        elif self.output_activation == 'sigmoid':
            x = tf.nn.sigmoid(x)
        elif self.output_activation == 'none':
            pass
        else:
            assert self.output_activation == False

        return x

class GlobalUpdate(snt.Module):

    def __init__(self, input_size, output_size, layer_sizes=None, output_activation=False, 
                       w_init=None, b_init=None, name=None):
        super(GlobalUpdate, self).__init__(name=name)
        self.input_size, self.output_size = input_size, output_size
        self.output_activation = output_activation

        layer_sizes = [output_size] if layer_sizes is None else layer_sizes
        self.n_layers = len(layer_sizes)

        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(snt.Linear(layer_sizes[i], 
                                          w_init=w_init,
                                          b_init=b_init, 
                                          name='linear{0:03d}'.format(i)))
        self.layers.append(snt.Linear(output_size, 
                                      w_init=w_init,
                                      b_init=b_init, 
                                      name='linear_out'))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = tf.nn.leaky_relu(x)

        x = self.layers[-1](x)
        if self.output_activation == 'leaky_relu':
            x = tf.nn.leaky_relu(x)
        elif self.output_activation == 'relu':
            x = tf.nn.relu(x)
        elif self.output_activation == 'softplus':
            x = tf.nn.softplus(x)
        elif self.output_activation == 'sigmoid':
            x = tf.nn.sigmoid(x)
        elif self.output_activation == 'none':
            pass
        else:
            assert self.output_activation == False

        return x

