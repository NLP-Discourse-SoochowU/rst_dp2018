# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018/4/2
@Description:
"""
from config import *
import numpy as np
import random
from utils.file_util import *

def generate_sample(edus_ids_list):
    """
    对padding好的EDU_list进行抽样和训练数据的准备
    :param edus_ids_list:
    :return:
    """
    center_sent = np.zeros(shape=(1, PAD_SIZE), dtype=np.int32)
    unk_edu_ids = np.ones(shape=(1, PAD_SIZE))
    # 存放；两个正样例， n个负样
    target_sents = np.zeros(shape=(NUM_SAMPLED + 2, PAD_SIZE), dtype=np.int32)
    target_sents_tag = np.zeros(NUM_SAMPLED + 2, dtype=np.int32)

    for index, sent in enumerate(edus_ids_list):
        center_sent[0] = edus_ids_list[index]
        if not sent.any():
            continue
        if index == 0:
            target_sents[0] = unk_edu_ids  # edi_ids中0位置放置的是全0向量
        else:
            target_sents[0] = edus_ids_list[index - 1]
        if index == (len(edus_ids_list) - 1):
            target_sents[1] = unk_edu_ids
        else:
            target_sents[1] = edus_ids_list[index + 1]
        # sents_ids.shape[0]就是所有论元的个数，从中随机抽取
        for idx_ in range(NUM_SAMPLED):
            tmp = edus_ids_list[np.random.randint(0, edus_ids_list.shape[0])]
            while not tmp.any():
                tmp = edus_ids_list[np.random.randint(0, edus_ids_list.shape[0])]
            target_sents[idx_ + 2] = tmp

        # 标志位
        target_sents_tag[0] = 1
        target_sents_tag[1] = 1  # 标志 正例

        yield center_sent, target_sents, target_sents_tag

def batch_gen(edus_ids_path):
    """
    生成一批训练数据
    :param sents_ids_path:
    :return:
    """
    edus_ids = load_data(edus_ids_path)
    # 数据打乱
    np.random.shuffle(edus_ids)
    # padding操作
    edus_ids = sents_padding(edus_ids, PAD_SIZE)
    single_gen = generate_sample(edus_ids)
    while True:
        center_batch = np.zeros([BATCH_SIZE_cbos, 1, PAD_SIZE], dtype=np.int32)  # 每一行存储上下文环境EDU对应下标
        target_batch = np.zeros([BATCH_SIZE_cbos, NUM_SAMPLED + 2, PAD_SIZE], dtype=np.int32)  # 存储负采样和正采样
        target_tag_batch = np.zeros([BATCH_SIZE_cbos, NUM_SAMPLED + 2], dtype=np.int32)
        for index in range(BATCH_SIZE_cbos):
            center_batch[index], target_batch[index], target_tag_batch[index] = next(single_gen)
        yield center_batch, target_batch, target_tag_batch

def sents_padding(sents_list, padding_size):
    """
    对给定的edu_ids数据进行padding操作
    :param sents_list: type list
    :param padding_size:
    :return: type numpy
    """
    padded_sents_list = None
    for line_ids in sents_list:
        temp_line = line_ids[:]
        if len(temp_line) > padding_size:
            temp_sent_ids = temp_line[:padding_size]
        else:
            temp_sent_ids = temp_line.copy()
            temp_sent_ids.extend([PAD_ids for _ in range(padding_size-len(temp_line))])
        # temp_sent_ids = np.array(temp_sent_ids)
        if padded_sents_list is None:
            padded_sents_list = [temp_sent_ids]
        else:
            padded_sents_list = np.append(padded_sents_list, [temp_sent_ids], axis=0)
    return padded_sents_list

if __name__ == "__main__":
    temp_ids = [[1, 2, 3], [idx for idx in range(60)]]
    print(sents_padding(temp_ids, 10))
