# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description:
"""
import torch
import torch.nn as nn
from config import *


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_sing = nn.Sequential(
            nn.Conv2d(
                in_channels=IN_CHANNEL_NUM,
                out_channels=FILTER_NUM,
                kernel_size=(FILTER_ROW, FILTER_COL),
                stride=STRIDE,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(KERNEL_ROW - 1, KERNEL_COL)),
        )

        self.conv_double = nn.Sequential(
            nn.Conv2d(
                in_channels=IN_CHANNEL_NUM,
                out_channels=FILTER_NUM,
                kernel_size=(FILTER_ROW * 2, FILTER_COL),
                stride=STRIDE,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(KERNEL_ROW - 2, KERNEL_COL)),
        )
        self.out = nn.Linear(FILTER_NUM, 2 * SPINN_HIDDEN)
        self.bi_out = nn.Linear(FILTER_NUM * 2, 2 * SPINN_HIDDEN)

    def forward(self, edu):
        edu_conv_sing = self.conv_sing(edu)
        edu_conv_doub = self.conv_double(edu)
        if Use_Bi_cnn:
            edu_conv = torch.cat((edu_conv_sing.view(edu_conv_sing.size(0), -1),
                                  edu_conv_doub.view(edu_conv_doub.size(0), -1),), 1)
            output = self.bi_out(edu_conv).squeeze()
        else:
            edu_conv = edu_conv_doub.view(edu_conv_doub.size(0), -1)
            output = self.out(edu_conv).squeeze()
        return output
