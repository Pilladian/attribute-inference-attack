# Python 3.8.5


import torch.nn as nn
import scipy.special
import numpy


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


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, lr):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.lr = lr

        self.activation_fn = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_fn(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_fn(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_inputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_fn(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_fn(final_inputs)

        return final_outputs
