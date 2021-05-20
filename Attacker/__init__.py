# Python 3.8.5


import torch.nn as nn


class MLP(nn.Module):

    def __init__(self,
                 n_features,
                 n_hidden,
                 n_classes,
                 activation,
                 dropout):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.lin1 = nn.Linear(n_features, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, n_classes)
        self.logso = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        h = self.lin1(inputs)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.lin2(h)
        h = self.activation(h)
        #h = self.dropout(h)
        h = self.lin3(h)
        out = self.logso(h)
        return out
