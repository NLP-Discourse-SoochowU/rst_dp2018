# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description: 对特征的使用
            Part-of-speech tags and words at the beginning and the end of the EDU
            Number of words of each EDU: 统计得到平均每个篇章包含的EDU个数为8.118个，最长EDU包含的token个数是50
                ，设置向量长度为50即可，将EDU包含单词个数作为下标。
            Number of words of each text span: 间接
            Number of EDUs of each text span：统计平均每个篇章包含的EDU个数是56个，考虑分区 (x<16, 16<=x<32, 30<=x<64, x>=64)
            Predicted relations of the two subtrees’ roots: 如果是EDU则rel会是None否则就是已经预测得到的关系
            Whether each text span or both text spans are included in one sentence: 根据标准树取即可
"""
from config import coarse2ids
from collections import deque
from config import SHIFT
from parser_model.rst_tree import rst_tree


class feature_generator:
    @staticmethod
    def generate_feat(tree_state):
        """
        对tree_state生成特征向量
        :param tree_state:
        :return:
        """
        stack_, buffer_ = tree_state
        r_ch = stack_[-1]
        l_ch = stack_[-2]
        f_edu = buffer_[0]
        if f_edu is None:
            # 对于空的edu情况，将feat1里面的word和pos都做PAD处理
            feat1_word_ids = [0, 0]
            feat1_pos_ids = [0, 0]
            feat2_ids = [0]
        else:
            # feat1 pos & word，否则调用相关函数获取对应id即可，理论上来说在生成树的时候这些信息直接存储了
            feat1_word_ids = [f_edu.temp_edu_ids[0], f_edu.temp_edu_ids[-1]]
            feat1_pos_ids = [f_edu.temp_pos_ids[0], f_edu.temp_pos_ids[-1]]

            # feat2 number of words of each edu，对EDU包含的token个数的下标统计
            feat2_ids = [len(f_edu.temp_edu_ids) - 1]

        # 对text_span的特征抽取
        if r_ch is None:
            feat3_r_ids = [0]
            feat4_r_ids = [0]
        else:
            # # feat. number of words of right text span
            # ...
            # feat3 number of edus of right text span
            count_ids = len(r_ch.temp_edu_ids)
            if count_ids < 16:
                feat3_r_ids = [1]
            elif count_ids < 32:
                feat3_r_ids = [2]
            elif count_ids < 64:
                feat3_r_ids = [3]
            else:
                feat3_r_ids = [4]
            # feat4 right text span 是否在一个句子内
            if judge_in_sent(r_ch):
                feat4_r_ids = [1]
            else:
                feat4_r_ids = [2]
        if l_ch is None:
            feat3_l_ids = [0]
            feat4_l_ids = [0]
        else:
            # # feat. number of words left text span
            # ...
            # feat3 number of edus of left text span
            count_ids = len(l_ch.temp_edu_ids)
            if count_ids < 16:
                feat3_l_ids = [1]
            elif count_ids < 32:
                feat3_l_ids = [2]
            elif count_ids < 64:
                feat3_l_ids = [3]
            else:
                feat3_l_ids = [4]

            # feat4 left text span 是否分别在一个句子内
            if judge_in_sent(l_ch):
                feat4_l_ids = [1]
            else:
                feat4_l_ids = [2]
        # feat5
        if r_ch is not None and l_ch is not None:
            # the first two elements in the stack: 是否同属一个句子
            if judge_in_sent(l_ch, r_ch):
                feat5_ids = [1]
            else:
                feat5_ids = [2]
            # relations of first two elements in the stack, 子树的根节点维护的relation是两个孩子的关系
            id_l = 19 if l_ch.child_rel is None else coarse2ids[l_ch.child_rel]
            id_r = 19 if r_ch.child_rel is None else coarse2ids[r_ch.child_rel]
            feat6_ids = [id_l, id_r]
        else:
            # 存在空的情况就直接将id设置为0, PADDING
            feat5_ids = [0]
            feat6_ids = [19, 19]
        return feat1_word_ids, feat1_pos_ids, feat2_ids, feat3_l_ids, feat3_r_ids, feat4_l_ids, feat4_r_ids, feat5_ids, \
               feat6_ids


def new_tree_state(tree):
    """
    将tree存成stack和buffer结构
    :param tree:
    :return:
    """
    # 初始化
    stack_ = deque()
    stack_.append(None)
    stack_.append(None)
    buffer_ = deque()
    buffer_.append(None)
    for edu_ in tree.edus:
        buffer_.appendleft(edu_)  # 对edu进行编码
    return stack_, buffer_


def forward_tree_state(tree_state, transition, rel=None):
    """
    负责根据当前指定的操作生成最新的状态的rst树.
    :param rel: 当前预测得到的孩子节点之间的关系
    :param tree_state: 当前栈和队列的状态
    :param transition: 当前将要执行的转移操作
    :return:
    """
    stack_, buffer_ = tree_state
    if transition == SHIFT:
        stack_.append(buffer_.popleft())
    else:
        r_ch = stack_.pop()
        l_ch = stack_.pop()
        parent_tree = rst_tree(l_ch=l_ch, r_ch=r_ch, child_rel=rel, temp_edu=l_ch.temp_edu + " " + r_ch.temp_edu,
                               temp_edu_span=(l_ch.temp_edu_span[0], r_ch.temp_edu_span[1]),
                               temp_edu_ids=l_ch.temp_edu_ids + r_ch.temp_edu_ids,
                               temp_pos_ids=l_ch.temp_pos_ids + r_ch.temp_pos_ids)
        stack_.append(parent_tree)
    return stack_, buffer_

def judge_in_sent(node, node_back=None):
    """
    判断一个树节点是否在一个完整的句子的内部，最好能在生成树的时候直接给这个参数进入
    :param node: 当前要判断的树节点
    :return:
    """
    ...
