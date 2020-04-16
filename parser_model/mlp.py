# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description:
"""
import torch.nn as nn
from torch.nn import functional as nnfunc
from config import *


class MLP(nn.Module):
    """
        多层感知器, 输入为hidden_size  不加入conn_tracker的情况
        输入->隐藏层->dropout->activate->output logits
    """

    def __init__(self, input_size=None, output_size=None, hidden_size=None, num_layers=None):
        nn.Module.__init__(self)
        if input_size is None:
            input_size = mlp_input_size
            hidden_size = mlp_hidden_size
            if num_layers is None:
                num_layers = mlp_num_layers
            if output_size is None:
                output_size = Transition_num
        # input to first hidden
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.input_dropout = nn.Dropout(p=mlp_dropout)
        self.input_activation = nn.ReLU()

        # multi hidden layers
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 2)])
        self.dropouts = nn.ModuleList([nn.Dropout(p=mlp_dropout) for _ in range(num_layers - 2)])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(num_layers - 2)])

        # probabilities
        self.logits = nn.Linear(hidden_size, output_size)

    def forward(self, input_values):
        hidden = self.input_linear(input_values)
        hidden = self.input_dropout(hidden)
        hidden = self.input_activation(hidden)
        # hidden layers
        for linear, dropout, activation in zip(self.linears, self.dropouts, self.activations):
            hidden = linear(hidden)
            hidden = dropout(hidden)
            hidden = activation(hidden)
        # sigmoid output layer
        output = self.logits(hidden)
        sig_output = nnfunc.sigmoid(output)
        return sig_output
