# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description:
"""
import torch
from config import *
from parser_model.spinn_model import SPINN
from parser_model.metric import Metrics
from parser_model.parser import Parser
from utils.file_util import *
from parser_model.tree_obj import tree_obj
from parser_model.feature_generator import feature_generator


def oracle(tree):
    """
         Desc: Back_traverse a gold tree of rst-dt
        Input:
               The tree object of rst_tree
       Output:
               Temp transition.
    """
    for node in tree.nodes:
        if node.left_child is not None and node.right_child is not None:
            if not Only_structure:
                yield REDUCE + "-" + node.child_NS_rel + "-" + node.child_rel
            else:
                if REL_CLS:
                    yield (REDUCE + "-" + node.child_NS_rel, node.child_rel)
                else:
                    yield REDUCE + "-" + node.child_NS_rel
        else:
            yield SHIFT


class SPINN_SR:
    """
        Training Container
    """
    def __init__(self):
        torch.manual_seed(SEED)
        # load rst vocabulary and cbos vector
        word2ids = load_data(VOC_WORD2IDS_PATH)
        pos2ids = load_data(POS_word2ids_PATH)
        if VERSION == 1:
            word_embed = load_data(CBOS_VEC_PATH)
        elif USE_Siamese:
            word_embed = load_data(VOC_VEC_Siamese_PATH)
            word2ids = load_data(VOC_WORD2IDS_Siamese_PATH)
        else:
            word_embed = load_data(VOC_VEC_PATH)

        # logger.log(logging.INFO, "loaded word embedding of vocabulary size %d" % len(vocab))
        # build the objective of SPINN
        self.model = SPINN(word2ids, pos2ids, word_embed)
        self.metric = Metrics()
        self.parser = Parser(self.model)
        self.feat_generator = feature_generator()

    def session(self, tree):
        return self.model.new_session(tree)

    def train_(self, trees, trees_eval_path=None):
        """
        train your own model
        :param trees:
        :param trees_eval_path:
        :return:
        """
        ...

    def evaluate(self, trees_eval_path, model):
        """
            Desc: 评测
            Input: rst_trees
            Output: P R F
        """
        print("不加特征评测...")
        trees_eval_gold = load_data(trees_eval_path)
        trees_eval = load_data(trees_eval_path)
        trees_pred = self.parser.parsing_all(trees_eval, model)
        trees_eval_pred = []
        for tree_ in trees_pred:
            trees_eval_pred.append(tree_obj(tree_))
        self.metric.eval_(trees_eval_gold, trees_eval_pred, self.model)
        print("评测结束!")

    def save(self, folder):
        """
            Desc: Save the model.
        """
        save_data(self.model, os.path.join(folder, "torch.bin"), append=True)
        save_data(self, os.path.join(folder, "model.pickle"), append=True)

    @staticmethod
    def restore(folder):
        """
            Desc: Load the model.
        """
        model = load_data(os.path.join(folder, "model.pickle"))
        model.model = load_data(os.path.join(folder, "torch.bin"))
        return model
