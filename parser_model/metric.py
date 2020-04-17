# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018/4/3
@Description:  parser_model
"""
import numpy as np
import torch
from utils.file_util import *
from config import MODELS2SAVE, VERSION, SET_of_version, REL_coarse2ids, Only_structure, REL_CLS


class Performance(object):
    def __init__(self, percision, recall):
        self.percision = percision
        self.recall = recall


class Metrics(object):
    def __init__(self):
        if not Only_structure:
            self.levels = ['span', 'nuclearity', 'relation']
        else:
            self.levels = ['span', 'nuclearity']
        self.span_perf = []
        self.nuc_perf = []
        self.rela_perf = []
        self.span_max = 0
        self.nucleas_max = 0
        self.relation_max = 0

    def eval_(self, goldtrees, predtrees, model):
        self.span_perf = []
        self.nuc_perf = []
        self.rela_perf = []
        for idx in range(len(goldtrees)):
            tmp_goldspan_ids, _ = self.get_span(goldtrees[idx])
            tmp_predspan_ids, _ = self.get_span(predtrees[idx])
            tmp_goldspan_NS_ids, _ = self.get_span_ns(goldtrees[idx])
            tmp_predspan_NS_ids, _ = self.get_span_ns(predtrees[idx])
            if not Only_structure or REL_CLS:
                tmp_goldspan_Rel_ids = self.get_span_rel_ids(goldtrees[idx])
                tmp_predspan_Rel_ids = self.get_span_rel_ids(predtrees[idx])
            else:
                tmp_goldspan_Rel_ids = None
                tmp_predspan_Rel_ids = None
            self.eval_one(tmp_goldspan_ids, tmp_predspan_ids, tmp_goldspan_NS_ids, tmp_predspan_NS_ids,
                          tmp_goldspan_Rel_ids, tmp_predspan_Rel_ids)
        self.report(model=model)

    def eval_one(self, tmp_goldspan_ids, tmp_predspan_ids, tmp_goldspan_ns_ids, tmp_predspan_ns_ids,
                 tmp_goldspan_rel_ids, tmp_predspan_rel_ids):
        """ compute the number of span in gold and pred for F and P.
        """
        # span
        allspan = [span for span in tmp_goldspan_ids if span in tmp_predspan_ids]
        allspan_gold_idx = [tmp_goldspan_ids.index(span) for span in allspan]
        allspan_pred_idx = [tmp_predspan_ids.index(span) for span in allspan]
        # ns
        all_goldspan_NS = [tmp_goldspan_ns_ids[idx] for idx in allspan_gold_idx]
        all_predspan_NS = [tmp_predspan_ns_ids[idx] for idx in allspan_pred_idx]
        # rel
        if not Only_structure or REL_CLS:
            all_goldspan_REL = [tmp_goldspan_rel_ids[idx] for idx in allspan_gold_idx]
            all_predspan_REL = [tmp_predspan_rel_ids[idx] for idx in allspan_pred_idx]
        else:
            all_goldspan_REL = all_predspan_REL = None

        p_1, r_1 = 0.0, 0.0
        for span in allspan:
            if span in tmp_goldspan_ids:
                p_1 += 1.0
            if span in tmp_predspan_ids:
                r_1 += 1.0
        p = (p_1 - 1) / (len(tmp_goldspan_ids) - 1)
        # r = (r_1 - 1) / (len(tmp_predspan_ids) - 1)
        self.span_perf.append(p)

        allspan_NS_count = sum(np.equal(all_goldspan_NS, all_predspan_NS))
        p = (allspan_NS_count - 1) / (len(tmp_goldspan_ids) - 1)
        self.nuc_perf.append(p)

        if all_goldspan_REL is not None:
            allspan_REL_count = sum(np.equal(all_goldspan_REL, all_predspan_REL))
            p = (allspan_REL_count - 1) / (len(tmp_goldspan_ids) - 1)
            self.rela_perf.append(p)

    def report(self, model):
        p_span = np.array(self.span_perf).mean()
        span_file_name = "/span_max_model.pth"
        p_ns = np.array(self.nuc_perf).mean()
        ns_file_name = "/nucl_max_model.pth"
        p_rel = np.array(self.rela_perf).mean()
        rel_file_name = "/rel_max_model.pth"

        if p_span > self.span_max:
            self.span_max = p_span
            self.save_model(file_name=span_file_name, model=model)
        print('F1 score on span level is: ', p_span)

        if p_ns > self.nucleas_max:
            self.nucleas_max = p_ns
            self.save_model(file_name=ns_file_name, model=model)
        print('F1 score on nuclearity level is: ', p_ns)

        if not Only_structure or REL_CLS:
            if p_rel > self.relation_max:
                self.relation_max = p_rel
                self.save_model(file_name=rel_file_name, model=model)
            print('F1 score on relation level is: ', p_rel)
        else:
            print('F1 score on relation level is: *')

    @staticmethod
    def save_model(file_name, model):
        dir2save = MODELS2SAVE + "/v" + str(VERSION) + "_set" + str(SET_of_version)
        safe_mkdir(dir2save)
        save_path = dir2save + file_name
        torch.save(model.state_dict(), save_path)

    @staticmethod
    def get_span(tree_):
        count_edus = 0
        span_ids = []
        for node in tree_.nodes:
            if node.left_child is None and node.right_child is None:
                count_edus += 1
            span_ids.append(node.temp_edu_span)
        return span_ids, count_edus

    @staticmethod
    def get_span_ns(tree_):
        ns_dict = {"Satellite": 0, "Nucleus": 1, "Root": 2}
        count_edus = 0
        span_ns_ids = []
        for node in tree_.nodes:
            if node.left_child is None and node.right_child is None:
                count_edus += 1
            span_ns_ids.append(ns_dict[node.type])
        return span_ns_ids, count_edus

    @staticmethod
    def get_span_rel_ids(tree_):
        coarse2ids = load_data(REL_coarse2ids)
        span_rel_ids = []
        for node in tree_.nodes:
            span_rel_ids.append(coarse2ids[node.rel])
        return span_rel_ids
