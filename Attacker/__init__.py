# Python 3.8.5


import torch.nn as nn


class MLP(nn.Module):

    def __init__(self,
                 n_features,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(nn.Linear(n_features, n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        # output layer
        self.layers.append(nn.Linear(n_hidden, n_classes))

    def forward(self, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
