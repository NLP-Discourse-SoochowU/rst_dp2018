# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018/5/4
@Description:
"""

from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as Var
from torch.nn import functional as nnfunc
from config import *
from utils.file_util import *
from parser_model.trackers import Tracker, Conn_Tracker
from parser_model.reducer import Reducer
from parser_model.mlp import MLP
from parser_model.cnn import CNN

_UNK = '<UNK>'
_DUMB = '<DUMB>'
_DUMB_IDX = 0


class SPINN(nn.Module):
    SHIFT = "SHIFT"
    REDUCE = "REDUCE"

    def __init__(self, word2ids, pos2ids, wordemb_weights):
        nn.Module.__init__(self)
        self.hidden_size = SPINN_HIDDEN
        self.wordemb_size = EMBED_SIZE
        self.posemb_size = POS_EMBED_SIZE
        self.connemb_size = CONN_EMBED_SIZE
        self.feature_emb_size = FEATURE_EMBED_SIZE
        self.word2ids = word2ids
        self.wordemb = nn.Embedding(len(word2ids), self.wordemb_size)
        self.wordemb.weight.data.copy_(torch.from_numpy(wordemb_weights))
        if word_emb_learn:
            self.wordemb.requires_grad = True
        else:
            self.wordemb.requires_grad = False
        self.pos2ids = pos2ids
        self.posemb = nn.Embedding(len(pos2ids.keys()), self.posemb_size)  # 随机初始化pos_embed
        self.posemb.requires_grad = True

        self.feat2_emb = nn.Embedding(52, FEAT2_SIZE)
        self.feat2_emb.requires_grad = True

        self.feat3_l_emb = nn.Embedding(5, FEAT3_SIZE)
        self.feat3_l_emb.requires_grad = True
        self.feat3_r_emb = nn.Embedding(5, FEAT3_SIZE)  # 对text_span包含的EDU个数向量化
        self.feat3_r_emb.requires_grad = True

        self.feat4_l_emb = nn.Embedding(3, FEAT4_SIZE)
        self.feat4_l_emb.requires_grad = True
        self.feat4_r_emb = nn.Embedding(3, FEAT4_SIZE)  # 对text_span是否在一个句子内部向量化
        self.feat4_r_emb.requires_grad = True

        self.feat5_emb = nn.Embedding(3, FEAT5_SIZE)  # 对两个text_span是否在一个句子里向量化
        self.feat5_emb.requires_grad = True
        self.feat6_emb = nn.Embedding(20, FEAT6_SIZE)  # 对已经预测得到的子树的关系向量化
        self.feat6_emb.requires_grad = True

        self.conn2ids = load_data(CONN_word2ids)
        self.connemb = nn.Embedding(len(self.conn2ids.keys()), self.connemb_size)  # 随机初始化pos_embed
        self.connemb.requires_grad = True

        self.feature2ids = edu_feature2ids
        self.feat_emb = nn.Embedding(len(self.feature2ids.keys()), self.feature_emb_size)  # 随机初始化
        self.feat_emb.requires_grad = True

        self.tracker = Tracker(self.hidden_size)
        self.conn_tracker = Conn_Tracker(self.hidden_size)
        self.reducer = Reducer(self.hidden_size)
        self.cnn = CNN()  # cnn
        self.edu_proj1 = nn.Linear(self.wordemb_size, self.hidden_size * 2)
        self.edu_proj2 = nn.Linear(self.wordemb_size * 3 + self.posemb_size * 3, self.hidden_size * 2)
        self.edu_proj3 = nn.Linear(self.wordemb_size * 3, self.hidden_size * 2)
        # bow加入
        self.edu_proj2_bow = nn.Linear(self.wordemb_size * 3 + self.posemb_size * 3 + self.wordemb_size,
                                       self.hidden_size * 2)
        self.edu_proj3_bow = nn.Linear(self.wordemb_size * 3 + self.wordemb_size, self.hidden_size * 2)
        # 为3个词的方案创建projection
        self.edu_proj4 = nn.Linear(self.wordemb_size * 4 + self.posemb_size * 4, self.hidden_size * 2)
        self.edu_proj5 = nn.Linear(self.wordemb_size * 4, self.hidden_size * 2)
        self.edu_proj6 = nn.Linear(self.wordemb_size * 4 + self.posemb_size * 4 + self.wordemb_size,
                                   self.hidden_size * 2)
        self.edu_proj7 = nn.Linear(self.wordemb_size * 4 + self.wordemb_size, self.hidden_size * 2)
        self.edu_proj8 = nn.Linear(self.wordemb_size * 2 + self.posemb_size + self.hidden_size * 2,
                                   self.hidden_size * 2)
        # feature
        input_size9 = self.wordemb_size * 4 + self.posemb_size * 4 + self.feature_emb_size + self.wordemb_size * 3
        hidden_size9 = int(input_size9 / 2)
        self.edu_proj9 = MLP(input_size=input_size9, output_size=2 * self.hidden_size, hidden_size=hidden_size9,
                             num_layers=2)

        input_size10 = self.wordemb_size * 4 + self.posemb_size * 4 + self.feature_emb_size + self.wordemb_size * 3 + \
                       self.hidden_size * 2
        hidden_size10 = int(input_size10 / 2)
        self.edu_proj10 = MLP(input_size=input_size10, output_size=2 * self.hidden_size, hidden_size=hidden_size10,
                              num_layers=2)
        # self.edu_proj9 = nn.Linear(self.wordemb_size * 4 + self.posemb_size * 4 + self.feature_emb_size +
        #                            self.wordemb_size * 3, self.hidden_size * 2)

        self.mlp_structure = MLP(output_size=Transition_num, num_layers=2)  # 对tracking的隐藏层输出进入mlp
        # self.mlp_structure = nn.Linear(mlp_input_size, Transition_num)

        # 针对关系集合创建, 创建3层感知器，因为关系类别多。
        self.mlp_rel = MLP(output_size=COARSE_REL_NUM, num_layers=2)

        # bilstm + attention
        edu_rnn_encoder_size = self.hidden_size * 2
        self.edu_rnn_encoder = nn.LSTM(self.wordemb_size + self.posemb_size, edu_rnn_encoder_size // 2,
                                       bidirectional=True)
        self.edu_attn_query = nn.Parameter(torch.randn(edu_rnn_encoder_size))
        self.edu_attn = nn.Sequential(
            nn.Linear(edu_rnn_encoder_size, edu_rnn_encoder_size),
            nn.Tanh()
        )

    @staticmethod
    def copy_session(session):
        """
        Desc: return a copy of a session.
        :param session:
        :return:
        """
        stack_, buffer_, tracking, conn_buffer, conn_tracking = session
        stack_clone = [s.clone() for s in stack_]
        buffer_clone = deque([b.clone() for b in buffer_])
        conn_buffer_clone = deque([b_.clone() for b_ in conn_buffer])
        h, c = tracking
        tracking_clone = h.clone(), c.clone()
        h_c, c_c = conn_tracking
        conn_tracking_clone = h_c.clone(), c_c.clone()
        return stack_clone, buffer_clone, tracking_clone, conn_buffer_clone, conn_tracking_clone

    @staticmethod
    def pad_edu(edu_ids=None, pos_ids=None):
        edu_ids_list = edu_ids[:]
        if pos_ids is not None:
            pos_ids_list = pos_ids[:]
        while len(edu_ids_list) < PAD_SIZE:
            edu_ids_list = np.append(edu_ids_list, PAD_ids)
            if pos_ids is not None:
                pos_ids_list = np.append(pos_ids_list, PAD_ids)
        if pos_ids is not None:
            return Var(torch.LongTensor(edu_ids_list)), Var(torch.LongTensor(pos_ids_list))
        else:
            return Var(torch.LongTensor(edu_ids_list))

    def generate_feat_vec(self, features=None):
        """
        对之前的特征的ids转换成对应向量 返回328
        :param features:
        :return:
        """
        feat1_word_ids, feat1_pos_ids, feat2_ids, feat3_l_ids, feat3_r_ids, feat4_l_ids, feat4_r_ids, feat5_ids, \
            feat6_ids = features
        feat1_word_embed = self.wordemb(torch.LongTensor(feat1_word_ids))  # (50, 100)
        feat1_pos_embed = self.posemb(torch.LongTensor(feat1_pos_ids))
        feat2_embed = self.feat2_emb(torch.LongTensor(feat2_ids))
        feat3_l_embed = self.feat3_l_emb(torch.LongTensor(feat3_l_ids))
        feat3_r_embed = self.feat3_r_emb(torch.LongTensor(feat3_r_ids))
        feat4_l_embed = self.feat4_l_emb(torch.LongTensor(feat4_l_ids))
        feat4_r_embed = self.feat4_r_emb(torch.LongTensor(feat4_r_ids))
        feat5_embed = self.feat5_emb(torch.LongTensor(feat5_ids))
        feat6_embed = self.feat6_emb(torch.LongTensor(feat6_ids))
        feat_vec = torch.cat([feat1_word_embed[0], feat1_word_embed[-1], feat1_pos_embed[0], feat1_pos_embed[-1],
                              feat2_embed[0], feat3_l_embed[0], feat3_r_embed[0], feat4_l_embed[0], feat4_r_embed[0],
                              feat5_embed[0], feat6_embed[0], feat6_embed[-1]]).view(1, -1)
        return feat_vec

    def new_session(self, tree):
        """
        Desc: Create a new session
        Input: the root of a new tree
        Output: stack, buffer, tracking
        :param tree:
        :return:
        """
        # 初始状态空栈中存在两个空数据
        stack = [Var(torch.zeros(self.hidden_size * 2)) for _ in range(2)]  # [dumb, dumb]
        # 初始化队列
        buffer_ = deque()
        conn_buffer = deque()

        for edu_ in tree.edus:
            buffer_.append(self.edu_encode(edu_))  # 对edu进行编码
            conn_buffer.append(self.get_edu_conn_vecs(edu_.temp_edu_conn_ids))

        buffer_.append(Var(torch.zeros(self.hidden_size * 2)))  # [edu, edu, ..., dumb]
        conn_buffer.append(Var(torch.zeros(1, CONN_EMBED_SIZE)))
        tracker_init_state = (Var(torch.zeros(1, self.hidden_size)), Var(torch.zeros(1, self.hidden_size)))
        tracking = self.tracker(stack, buffer_, tracker_init_state)  # forward of Tracker
        conn_tracker_init_state = (Var(torch.zeros(1, self.hidden_size)), Var(torch.zeros(1, self.hidden_size)))
        conn_tracking = self.conn_tracker(conn_buffer, conn_tracker_init_state)
        return stack, buffer_, tracking, conn_buffer, conn_tracking

    def score(self, session, features=None):
        """
        Desc: sigmoid(fullc(h->1))
            使用BCE loss的时候返回一个概率，用sigmoid
            使用Cross entropy loss的时候返回一组概率值，个数和标签数一致
        :param features:
        :param session:
        :return:
        """
        _, _, tracking, _, conn_tracking = session
        h, _ = tracking
        c_h, _ = conn_tracking
        if USE_FEATURE:
            feat_vec = self.generate_feat_vec(features=features)
            if USE_conn_tracker:
                tracker_h = torch.cat((h, c_h, feat_vec), 1)
            else:
                tracker_h = torch.cat((h, feat_vec), 1)
        else:
            if USE_conn_tracker:
                tracker_h = torch.cat((h, c_h), 1)
            else:
                tracker_h = h
        score_output = self.mlp_structure(tracker_h)  # (1, 584)
        return score_output

    def score_rel(self, session, features=None):
        """
        Desc: sigmoid(fullc(h->1))
            使用BCE loss的时候返回一个概率，用sigmoid
            使用Cross entropy loss的时候返回一组概率值，个数和标签数一致
        :param features:
        :param session:
        :return:
        """
        _, _, tracking, _, conn_tracking = session
        h, _ = tracking
        c_h, _ = conn_tracking
        if USE_FEATURE:
            feat_vec = self.generate_feat_vec(features=features)
            if USE_conn_tracker:
                tracker_h = torch.cat((h, c_h, feat_vec), 1)
            else:
                tracker_h = torch.cat((h, feat_vec), 1)
        else:
            if USE_conn_tracker:
                tracker_h = torch.cat((h, c_h), 1)
            else:
                tracker_h = h
        score_output = self.mlp_rel(tracker_h)
        return score_output

    def get_edu_conn_vecs(self, temp_edu_conn_ids):
        ids = Var(torch.LongTensor(np.array(temp_edu_conn_ids)))
        emb = self.connemb(ids)
        return emb

    def edu_bilstm_encode(self, word_emb, tags_emb):
        inputs = torch.cat([word_emb, tags_emb], 1).unsqueeze(1)  # (seq_len, batch, input_size)
        hs, _ = self.edu_rnn_encoder(inputs)  # hs.size()  (seq_len, batch, hidden_size)
        hs = hs.squeeze()  # size: (seq_len, hidden_size)
        keys = self.edu_attn(hs)  # size: (seq_len, hidden_size)
        # print(keys.size())
        # print(self.edu_att_query.size())
        # input(keys.matmul(self.edu_attn_query).size())  # change 2
        attn = nnfunc.softmax(keys.matmul(self.edu_attn_query), 0)
        output = (hs * attn.view(-1, 1)).sum(0)
        # input("bi lstm!")
        return output, attn

    def edu_encode(self, edu):
        # print(edu.temp_edu_ids)
        # input(edu.temp_edu)
        """
        Desc: Use the 0 1 -1 word vector of a sentence to encode an EDU
        Input: An object of rst_tree, leaf node
        Output: An output of code with lower dimension.
        :param edu:
        :return:
        """
        edu_ids = edu.temp_edu_ids[:]
        pos_ids = edu.temp_pos_ids[:]
        if len(edu_ids) == 0:
            return torch.zeros(self.hidden_size * 2)
        if VERSION == 1:
            proj_out = ...
        # 2 + 1
        elif VERSION == 2:
            proj_out = ...
        # encode the EDU with CNN
        elif VERSION == 3:
            proj_out = ...
        # 3 + 1
        elif VERSION == 4:
            proj_out = ...
        elif VERSION == 5:
            # VERSION == 5 bilstm + self attention
            word_emb = self.wordemb(torch.LongTensor(edu_ids))  # (50, 100)
            pos_embed = self.posemb(torch.LongTensor(pos_ids))
            # print(word_emb.size(), "=======", pos_embed.size())
            # input()
            rnn_emb, attn = self.edu_bilstm_encode(word_emb, pos_embed)
            edu_embed = torch.cat([word_emb[0], word_emb[-1], pos_embed[0], rnn_emb])
            # project
            proj_out = self.edu_proj8(edu_embed)
            # input(rnn_emb.size())
            # proj_out = rnn_emb
        elif VERSION == 6:
            proj_out = ...
        elif VERSION == 7:
            proj_out = ...
        else:
            proj_out = ...
        return proj_out

    def get_words_tag_ids(self, edu_ids, pos_ids):
        if len(edu_ids) == 1:
            w1 = edu_ids[0]
            w2 = self.word2ids[PAD]
            w3 = self.word2ids[PAD]
            w_1 = self.word2ids[PAD]
            p1 = pos_ids[0]
            p2 = self.pos2ids[PAD]
            p3 = self.pos2ids[PAD]
            p_1 = self.pos2ids[PAD]
        elif len(edu_ids) == 2:
            w1 = edu_ids[0]
            w2 = edu_ids[1]
            w3 = self.word2ids[PAD]
            w_1 = self.word2ids[PAD]
            p1 = pos_ids[0]
            p2 = pos_ids[1]
            p3 = self.pos2ids[PAD]
            p_1 = self.pos2ids[PAD]
        else:
            w1 = edu_ids[0]
            w2 = edu_ids[1]
            w3 = edu_ids[2]
            w_1 = edu_ids[-1]
            p1 = pos_ids[0]
            p2 = pos_ids[1]
            p3 = pos_ids[2]
            p_1 = pos_ids[-1]
        if VERSION == 4 or VERSION == 6 or VERSION == 7 or VERSION == 8:
            return [w1, w2, w3, w_1], [p1, p2, p3, p_1]
        elif VERSION == 2:
            return [w1, w2, w_1], [p1, p2, p_1]

    def forward(self, session, transition):
        """
        Desc: The forward of SPINN
        Input:
               session and (shift or reduce)
        output:
               newest stack and buffer, lstm output
        :param session:
        :param transition:
        :return:
        """
        stack_, buffer_, tracking, conn_buffer, conn_tracking = session
        if transition == SHIFT:
            stack_.append(buffer_.popleft())
        else:
            s1 = stack_.pop()
            s2 = stack_.pop()
            compose = self.reducer(s2, s1, tracking, conn_tracking)  # the forward of Reducer
            stack_.append(compose)
        # 最新状态转移
        tracking = self.tracker(stack_, buffer_, tracking)  # The forward of the Tracker
        conn_tracking = self.conn_tracker(conn_buffer, conn_tracking)
        if transition == SHIFT:
            conn_buffer.popleft()
        return stack_, buffer_, tracking, conn_buffer, conn_tracking
