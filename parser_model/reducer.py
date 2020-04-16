# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description:
"""
import torch
import torch.nn as nn
from config import *


class Reducer(nn.Module):
    """
    Desc： The composition function for reduce option.
    """

    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.proj_tracking_all = nn.Linear(self.hidden_size * 4, self.hidden_size * 5)
        self.proj_tracking = nn.Linear(self.hidden_size * 3, self.hidden_size * 5)
        self.proj = nn.Linear(self.hidden_size * 2, self.hidden_size * 5)

    def forward(self, left, right, tracking, conn_tracking):
        """
        Desc: The forward of Reducer
        input: The rep of left node and right node, e is the tree lstm's output, it has a different D.
        output:
               The rep of temp node
        :param conn_tracking:
        :param left:
        :param right:
        :param tracking:
        :return:
        """
        h1, c1 = left.chunk(2)
        h2, c2 = right.chunk(2)
        e_h, e_c = tracking
        c_h, c_c = conn_tracking
        if USE_tracking and USE_conn_tracker:
            g, i, f1, f2, o = self.proj_tracking_all(torch.cat([h1, h2, e_h.squeeze(), c_h.squeeze()])).chunk(5)
        elif USE_tracking:
            g, i, f1, f2, o = self.proj_tracking(torch.cat([h1, h2, e_h.squeeze()])).chunk(5)
        elif USE_conn_tracker:
            g, i, f1, f2, o = self.proj_tracking(torch.cat([h1, h2, c_h.squeeze()])).chunk(5)
        else:
            g, i, f1, f2, o = self.proj(torch.cat([h1, h2])).chunk(5)
        c = g.tanh() * i.sigmoid() + f1.sigmoid() * c1 + f2.sigmoid() * c2
        h = o.sigmoid() * c.tanh()
        return torch.cat([h, c])
