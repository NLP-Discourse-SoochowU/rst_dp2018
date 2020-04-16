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
    """，
        desc: 考虑spinn编码针对一个EDU进行编码，EDU表示采用(1, 1, EDU_length, Embedding_size)
        conv
        FILTER_NUM = 2  考虑训练数据缺少，当前filter个数根据情况累加
        FILTER_ROW = 1
        FILTER_COL = 2
        STRIDE = 1

        padding
        kernel_size = (2, 2)  对 Bi_gram 的使用
    """

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
        # 根据pool的结果进行view转换之后确定out的input size
        self.out = nn.Linear(FILTER_NUM, 2 * SPINN_HIDDEN)
        self.bi_out = nn.Linear(FILTER_NUM * 2, 2 * SPINN_HIDDEN)

    def forward(self, edu):
        # print(edu.size())
        edu_conv_sing = self.conv_sing(edu)
        edu_conv_doub = self.conv_double(edu)
        # print(edu_conv_doub.size())
        # input("=====")
        if Use_Bi_cnn:
            edu_conv = torch.cat((edu_conv_sing.view(edu_conv_sing.size(0), -1),
                                  edu_conv_doub.view(edu_conv_doub.size(0), -1),), 1)
            output = self.bi_out(edu_conv).squeeze()
        else:
            edu_conv = edu_conv_doub.view(edu_conv_doub.size(0), -1)
            output = self.out(edu_conv).squeeze()
        return output
