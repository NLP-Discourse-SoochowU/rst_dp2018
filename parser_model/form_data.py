# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018/4/5
@Description:
"""
from utils.file_util import *
from config import *
from parser_model.rst_tree import rst_tree
from parser_model.tree_obj import tree_obj
from utils.rst_utils import get_edus_info
from stanfordcorenlp import StanfordCoreNLP

path_to_jar = 'stanford-corenlp-full-2018-02-27'

"""
path_to_jar = 'path_to/stanford-parser-full-2014-08-27/stanford-parser.jar'
path_to_models_jar = 'path_to/stanford-parser-full-2014-08-27/stanford-parser-3.4.1-models.jar'

dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

result = dependency_parser.raw_parse('I shot an elephant in my sleep')
dep = result.next()

list(dep.triples())
"""


class form_data:
    def __init__(self):
        # self.path=path
        self.train_data = dict()
        self.edu_stack = []
        self.edu_queue = []
        self.root = None
        # tree list
        self.trees_list = []
        self.train_data_list = []
        self.train_label_list = []
        self.total_file_num = 0
        self.total_edu_num = 0
        # feat
        self.feat = None
        self.edu_node = []
        self.nlp = StanfordCoreNLP(path_to_jar)

    def traverse_data_label(self, root):
        if root is None:
            return
        else:
            self.traverse_data_label(root.left_child)
            self.traverse_data_label(root.right_child)
            self.get_features()
            if root.left_child is None and root.right_child is None:
                q_head = self.edu_queue.pop(0)
                self.edu_stack.append(q_head)
                temp_transition = "Shift"
            else:
                node_NS = root.child_NS_rel
                node_Rel = root.child_rel
                self.edu_stack.pop()
                self.edu_stack.pop()
                self.edu_stack.append(root)
                temp_transition = "Reduce-" + node_NS + "-" + node_Rel
            self.train_label_list.append(temp_transition)
            self.train_data_list.append(self.feat)

    def get_features(self):
        queue_state = self.edu_queue[0] if len(self.edu_queue) > 0 else None
        stack_first_state = self.edu_stack[-1] if len(self.edu_stack) > 0 else None
        stack_second_state = self.edu_stack[-2] if len(self.edu_stack) > 1 else None
        self.feat = (stack_first_state, stack_second_state, queue_state)

    def init_queue_stack(self):
        self.edu_stack = []
        self.edu_queue = []
        for edu_node in self.edu_node:
            self.edu_queue.append(edu_node)

    def form_trees(self, type_):
        if type_ == "train":
            self.form_trees_type_("train")
        if type_ == "test":
            self.form_trees_type_("test")

    def form_trees_type_(self, type_):
        self.trees_list = []
        rst_dt_path = RST_DT_TRAIN_PATH if type_ is "train" else RST_DT_TEST_PATH
        if not USE_Siamese:
            save_tree_path = RST_TRAIN_TREES if type_ is "train" else RST_TEST_TREES
        else:
            save_tree_path = RST_TRAIN_Siamese_TREES if type_ is "train" else RST_TEST_Siamese_TREES
        for file_name in os.listdir(rst_dt_path):
            if file_name.endswith('.out.dis'):
                self.root = self.build_one_tree(rst_dt_path, file_name)
                tree_obj_ = tree_obj(self.root)
                self.trees_list.append(tree_obj_)
        # save trees
        save_data(self.trees_list, save_tree_path)

    def build_one_tree(self, rst_dt_path, file_name):
        """ build edu_list and edu_ids according to EDU files
        """
        print(file_name)
        temp_path = os.path.join(rst_dt_path, file_name)
        edus_list, edu_span_list, edus_ids_list, edus_tag_ids_list, edus_conns_list, edu_headword_ids_list,\
            edu_has_center_word_list = get_edus_info(temp_path.replace(".dis", ".edus"),
                                                     temp_path.replace(".out.dis", ".out"),
                                                     nlp=self.nlp, file_name=file_name)
        dis_tree_obj = open(temp_path, 'r')
        lines_list = dis_tree_obj.readlines()
        rel_raw2coarse = load_data(REL_raw2coarse)
        root = rst_tree(type_="Root", lines_list=lines_list, temp_line=lines_list[0], file_name=file_name, rel="span",
                        rel_raw2coarse=rel_raw2coarse)
        root.create_tree(temp_line_num=1, p_node_=self.root)
        edu_node = []

        root.config_edus(temp_node=self.root, temp_edu_list=edus_list, temp_edu_span_list=edu_span_list,
                         temp_eduids_list=edus_ids_list, edus_tag_ids_list=edus_tag_ids_list,
                         edu_node=edu_node, edus_conns_list=edus_conns_list,
                         edu_headword_ids_list=edu_headword_ids_list, edu_has_center_word_list=edu_has_center_word_list,
                         total_edus_num=len(edus_ids_list))
        return root

    def build_tree_obj_list(self):
        parse_trees = []
        for file_name in os.listdir(RAW_TXT):
            if file_name.endswith(".out"):
                tmp_edus_list = []
                sent_path = os.path.join(RAW_TXT, file_name)
                edu_path = sent_path + ".edu"
                edus_list, edu_span_list, edus_ids_list, edus_tag_ids_list, edus_conns_list, edu_headword_ids_list, \
                    edu_has_center_word_list = get_edus_info(edu_path, sent_path, nlp=self.nlp, file_name=file_name)
                for _ in range(len(edus_list)):
                    tmp_edu = rst_tree()
                    tmp_edu.temp_edu = edus_list.pop(0)
                    tmp_edu.temp_edu_span = edu_span_list.pop(0)
                    tmp_edu.temp_edu_ids = edus_ids_list.pop(0)
                    tmp_edu.temp_pos_ids = edus_tag_ids_list.pop(0)
                    tmp_edu.temp_edu_conn_ids = edus_conns_list.pop(0)
                    tmp_edu.temp_edu_heads = edu_headword_ids_list.pop(0)
                    tmp_edu.temp_edu_has_center_word = edu_has_center_word_list.pop(0)
                    tmp_edus_list.append(tmp_edu)
                tmp_tree_obj = tree_obj()
                tmp_tree_obj.file_name = file_name
                tmp_tree_obj.assign_edus(tmp_edus_list)
                parse_trees.append(tmp_tree_obj)
        return parse_trees
