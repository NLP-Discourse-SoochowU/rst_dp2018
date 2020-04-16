# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description:
"""
import torch
import torch.nn as nn
from config import CONN_EMBED_SIZE


class Tracker(nn.Module):
    """
        Desc: tracker for tree lstm
    """
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(3 * self.hidden_size, hidden_size)

    def forward(self, stack, buffer_, state):
        """
        Desc: tracking lstm
        :param stack:
        :param buffer_:
        :param state:
        :return:
        """
        s2, s1 = stack[-2], stack[-1]
        b1 = buffer_[0]
        s2h, s2c = s2.chunk(2)
        s1h, s1c = s1.chunk(2)
        b1h, b1c = b1.chunk(2)
        cell_input = torch.cat([s2h, s1h, b1h]).view(1, -1)  # 联合放入模式
        # state1, state2 = state
        tracking_h, tracking_c = self.rnn(cell_input, state)  # forward of the model rnn
        return tracking_h.view(1, -1), tracking_c.view(1, -1)


class Conn_Tracker(nn.Module):
    """
    Desc: tracker for connective in buffer of a discourse
    不采用拼接方式，单独采用来自buffer里面的连接词逐个作为输入 这也是 拆分conn到单词的原因
    """

    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell((CONN_EMBED_SIZE / 2), hidden_size)

    def forward(self, conn_buffer, state):
        """
        Desc: tracking lstm
        :param conn_buffer:
        :param state:
        :return:
        """
        b = conn_buffer[0]
        # 对b里面包含的若干连接词单词依次进入lstm
        for conn_emb in b:
            h, _ = conn_emb.chunk(2)
            cell_input = h.view(1, -1)  # 联合放入模式
            tracking_h, tracking_c = self.rnn(cell_input, state)  # forward of the model rnn
            state = tracking_h.view(1, -1), tracking_c.view(1, -1)  # 看初始化
        return state[0], state[1]
