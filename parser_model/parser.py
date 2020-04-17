# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018/4/3
@Description:  parser_model
"""
import random
import copy
from config import *
from utils.file_util import *
from parser_model.rst_tree import rst_tree
import numpy as np
from parser_model.feature_generator import *


class Parser:
    def __init__(self, model=None):
        # load the spinn model
        self.model = model
        self.tmp_edu = 0
        self.beam_search_session = self.beam_search_stack = self.beam_search_queue = self.beam_search_scores = None
        self.init_beam_search()
        self.feat_generator = feature_generator()

    def init_beam_search(self):
        # beam search表
        self.beam_search_session = []
        self.beam_search_stack = []
        self.beam_search_queue = []
        self.beam_search_scores = []

    def restore(self, folder):
        self.model = load_data(os.path.join(folder, "model.pickle"))
        self.model.model = load_data(os.path.join(folder, "torch.bin"))
        return self.model

    def parsing_all(self, trees_eval, model=None):
        if model is None:
            model = self.model
        else:
            ...
        parsed_tree_list = []
        for tree in trees_eval:
            if USE_beam_search:  # slow
                parsed_tree = self.parsing_beam(tree, model)
            else:
                parsed_tree = self.parsing(tree, model)
            parsed_tree_list.append(parsed_tree)
        return parsed_tree_list

    def parsing_beam(self, tree, model=None):
        """ Desc: parsing tree(tree_obj) of a given discourse.
        """
        if model is None:
            model = self.model
        session = model.new_session(tree)
        stack_ = []
        queue_ = tree.edus[:]
        self.beam_search_session.append(session)
        self.beam_search_stack.append(copy.deepcopy(stack_))
        self.beam_search_queue.append(copy.deepcopy(queue_))
        self.beam_search_scores.append(0)
        tmp_filename = tree.file_name
        while not self.parse_end():
            # create a new beam table
            tmp_beam_scores = []
            tmp_beam_sessions = []
            tmp_beam_stack = []
            tmp_beam_queue = []
            while len(self.beam_search_session) > 0:
                # print("============beam_search_session pop==========")
                tmp_score = self.beam_search_scores.pop(0)
                tmp_session = self.beam_search_session.pop(0)
                tmp_stack = self.beam_search_stack.pop(0)
                tmp_queue = self.beam_search_queue.pop(0)
                if not self.parsing_end(tmp_session):
                    predict_score = model.score(tmp_session)
                    # choose beam_size transitions
                    # print("A ============beam_search_session pop==========")
                    tmp_transitions, tmp_scores = self.choose_transitions_beam(predict_score, tmp_queue, tmp_stack)

                    # print("B ============beam_search_session pop==========")
                    for idx in range(len(tmp_transitions)):
                        cp_stack = copy.deepcopy(tmp_stack)
                        cp_queue = copy.deepcopy(tmp_queue)
                        cp_session = model.copy_session(tmp_session)
                        if tmp_transitions[idx] == SHIFT:
                            rel = None
                        else:
                            rel_score = model.score_rel(cp_session)
                            rel = self.choose_rel(rel_score)
                        self.form_tree(cp_stack, cp_queue, tmp_transitions[idx], rel=rel)

                        new_session = model(cp_session, tmp_transitions[idx])
                        tmp_beam_scores.append(tmp_score + np.log(tmp_scores[idx]))  # path score
                        tmp_beam_sessions.append(new_session)
                        tmp_beam_stack.append(cp_stack)
                        tmp_beam_queue.append(cp_queue)
                        # print("C ============beam_search_session pop==========")
                else:
                    """ save
                    """
                    tmp_beam_scores.append(tmp_score)
                    tmp_beam_sessions.append(tmp_session)
                    tmp_beam_stack.append(tmp_stack)
                    tmp_beam_queue.append(tmp_queue)

            # print("D ============beam_search_session pop==========")
            pop_beam_scores = tmp_beam_scores[:]
            for _ in range(BEAM_SIZE):
                tmp_max_score = np.max(pop_beam_scores)
                pop_idx = np.where(pop_beam_scores == tmp_max_score)[0][0]
                beam_idx = np.where(tmp_beam_scores == tmp_max_score)[0][0]
                self.beam_search_scores.append(pop_beam_scores.pop(pop_idx))
                self.beam_search_queue.append(tmp_beam_queue[beam_idx])
                self.beam_search_stack.append(tmp_beam_stack[beam_idx])
                self.beam_search_session.append(tmp_beam_sessions[beam_idx])
                # print("E ============beam_search_session pop==========")
        # end
        tree = self.beam_search_stack[0][-1]  # select the first one (highest score)
        tree.type = "Root"
        tree.rel = "span"
        tree.file_name = tmp_filename
        tree.config_nodes(tree)
        self.init_beam_search()
        return tree

    def parse_end(self):
        """ judge if all sessions are over in the beam table
        """
        for tmp_session in self.beam_search_session:
            if not self.parsing_end(tmp_session):
                return False
        return True

    def parsing(self, tree, model_=None):
        """ Desc: parsing tree(tree_obj) of a given discourse.
        """
        if model_ is None:
            model_ = self.model
        session = model_.new_session(tree)
        if USE_FEATURE:
            tree_state = new_tree_state(tree)
        stack_ = []
        queue_ = tree.edus[:]
        tmp_filename = tree.file_name
        while not self.parsing_end(session):
            if USE_FEATURE:
                features = self.feat_generator.generate_feat(tree_state)
                predict_score = model_.score(session, features)
            else:
                predict_score = model_.score(session)
            transition = self.choose_transition(predict_score, queue_, stack_)
            if transition == SHIFT:
                rel = None
            else:
                if USE_FEATURE:
                    features = self.feat_generator.generate_feat(tree_state)
                    rel_score = model_.score_rel(session, features)
                else:
                    rel_score = model_.score_rel(session)
                rel = self.choose_rel(rel_score)
            # form_tree
            self.form_tree(stack_, queue_, transition, rel)
            session = model_(session, transition)
            if USE_FEATURE:
                tree_state = forward_tree_state(tree_state, transition, rel)
        # end
        tree_ = stack_[-1]
        tree_.type = "Root"
        tree_.rel = "span"
        tree_.file_name = tmp_filename
        tree_.config_nodes(tree_)
        return tree_

    def choose_transitions_beam(self, score, queue_, stack_):
        """ beam_size: capacity and label num = 1:2 BEAM_SIZE
        """
        beam_scores = []
        beam_transition_idxes = []
        score = score.data.numpy()[0].tolist()
        score_tmp = copy.deepcopy(score)
        for _ in range(BEAM_SIZE):
            tmp_max_score = np.max(score_tmp)
            beam_scores.append(tmp_max_score)
            pop_idx = np.where(score_tmp == tmp_max_score)[0][0]
            transition_idx = np.where(score == tmp_max_score)[0][0]
            beam_transition_idxes.append(transition_idx)
            score_tmp.pop(pop_idx)

        beam_transitions = []
        for transition_idx in beam_transition_idxes:
            if transition_idx == 0 and len(queue_) > 0:
                transition = ids2action[0]
            elif transition_idx == 0 and len(queue_) == 0:
                transition = self.get_random_reduce("ns")
            elif transition_idx != 0 and len(stack_) < 2:
                transition = ids2action[0]
            else:
                transition = ids2action[transition_idx]
            beam_transitions.append(transition)
        return beam_transitions, beam_scores

    def parse(self, parse_model, parse_data):
        safe_mkdir(TREES_PARSED)
        trees_eval_ = parse_data[:]
        trees_list = self.parsing_all(trees_eval_, parse_model)
        # painting tree
        strtree_dict = dict()
        for tree in trees_list:
            strtree, draw_file = self.draw_one_tree(tree, TREES_PARSED)
            strtree_dict[draw_file] = strtree

        save_data(trees_list, TREES_PARSED + "/trees_list.pkl")
        save_data(strtree_dict, TREES_PARSED + "/strtrees_dict.pkl")

    def draw_one_tree(self, tree, path):
        self.tmp_edu = 1
        strtree = self.parse2strtree(tree)
        draw_file = path + "/" + tree.file_name.split(".")[0] + ".ps"
        return strtree, draw_file

    def parse2strtree(self, root):
        """
         ( NN-textualorganization ( SN-attribution ( EDU 1 )  ( NN-list ( EDU 2 )  ( EDU 3 )  )  )  ( EDU 5 )  )
        """
        if root.left_child is None:
            tmp_str = "( EDU " + str(self.tmp_edu) + " )"
            self.tmp_edu += 1
            return tmp_str
        else:
            tmp_str = ("( " + root.child_NS_rel) + "-" + root.child_rel + " " + self.parse2strtree(root.left_child) + \
                      "  " + self.parse2strtree(root.right_child) + "  )"
            return tmp_str

    @staticmethod
    def form_tree(stack_, queue_, transition, rel):
        type_dict = {"N": "Nucleus", "S": "Satellite"}
        if transition == SHIFT:
            edu_tmp = queue_.pop(0)
            stack_.append(edu_tmp)
        else:
            transition_split = transition.split("-")
            child_ns_rel = transition.split("-")[1]
            if not Only_structure:
                child_rel = "-".join(transition_split[2:])
            elif REL_CLS:
                child_rel = rel
            else:
                child_rel = "no_care"
            # 建树
            right_c = stack_.pop()
            left_c = stack_.pop()
            left_c.type = type_dict[child_ns_rel[0]]
            right_c.type = type_dict[child_ns_rel[1]]
            if child_ns_rel == "NN":
                left_c.rel = child_rel
                right_c.rel = child_rel
            else:
                left_c.rel = child_rel if left_c.type == "Satellite" else "span"
                right_c.rel = child_rel if right_c.type == "Satellite" else "span"
            new_tree_node = rst_tree(l_ch=left_c, r_ch=right_c, ch_ns_rel=child_ns_rel, child_rel=child_rel)
            stack_.append(new_tree_node)

    @staticmethod
    def clone_qs(q_s):
        q_s_ = [copy.copy(edu) for edu in q_s]
        return q_s_

    @staticmethod
    def parsing_end(session):
        stack_, buffer_, _, _, _ = session
        state_ = True if len(stack_) == 3 and len(buffer_) == 1 else False
        return state_

    @staticmethod
    def get_random_reduce(type_):
        if type_ == "ns":
            action_ids = random.randint(1, Transition_num - 1)
        else:
            action_ids = None
            input("wrong!")
        return ids2action[action_ids]

    @staticmethod
    def choose_rel(score_rel):
        score_rel = score_rel.data.numpy()[0]
        rel_idx = np.where(score_rel == np.max(score_rel))[0][0]
        rel = ids2coarse[rel_idx]
        return rel

    @staticmethod
    def choose_transition(score, queue_, stack_):
        score = score.data.numpy()[0]
        transition_idx = np.where(score == np.max(score))[0][0]
        if transition_idx == 0 and len(queue_) > 0:
            # shift
            transition = SHIFT
        elif transition_idx == 0:
            score[transition_idx] = 0
            transition_idx = np.where(score == np.max(score))[0][0]
            transition = ids2action[transition_idx]
        elif transition_idx != 0 and len(stack_) < 2:
            transition = SHIFT
        else:
            transition = ids2action[transition_idx]
        return transition
