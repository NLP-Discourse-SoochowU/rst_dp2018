# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018/4/5
@Description:   语料需要转换，转换前对需要结合fetures特征,所以能够根据当前树的状态，得到对应的doc对象才能获取对应的特征
                遍历过程中验证树建立的合法性，进行前序遍历
                1希望建立的树和文件保持一致
                2希望shift-reduce过程和configuration保持同步，shift-reduce过程和文件的后续遍历保持一致，验证。
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
        # 创建字典，c对应操作
        self.train_data = dict()
        self.edu_stack = []
        self.edu_queue = []
        self.root = None
        # tree list
        self.trees_list = []
        # 生成的训练数据
        self.train_data_list = []
        # 生成的训练标签
        self.train_label_list = []
        # 统计
        self.total_file_num = 0
        self.total_edu_num = 0
        # feat
        self.feat = None
        self.edu_node = []
        # 创建stanford parser
        self.nlp = StanfordCoreNLP(path_to_jar)

    def traverse_data_label(self, root):
        """
        后序遍历,进行序列提取(将当前stack和queue作为参数传递给函数进行特征提取)，操作标签提取
        提取merge文件
        情况1. 当前节点左右孩子都是空，作为edu看待
        #记录EDUs的个数
        N = len(doc.edudict)
        for idx in range(1, N+1, 1):
            node = SpanNode(prop=None)
            node.text = doc.edudict[idx]
            #span的边界确立,包含哪几个EDU形成的跨度
            node.eduspan, node.nucspan = (idx, idx), (idx, idx)
            #当前span关系的核EDU指向
            node.nucedu = idx
            #当前EDUs对应成RST节点进入队列
            self.Queue.append(node)

            队列 和 栈作为全局变量

            2017年12月20日发现问题，树和SpanNode数据不一致原因：segmentation得到的edu找回率非100%所以，goldTree的
        构建要通过对标准文件中的标准EDU构建对应merge文件，然后用这些生成文件替代原segmentation得到的文件。
        :param root:
        :return:
        """
        if root is None:
            return
        else:
            self.traverse_data_label(root.left_child)
            self.traverse_data_label(root.right_child)
            self.get_features()  # 获取当前configuration
            # shift操作
            if root.left_child is None and root.right_child is None:
                q_head = self.edu_queue.pop(0)
                self.edu_stack.append(q_head)
                temp_transition = "Shift"
            else:
                node_NS = root.child_NS_rel
                node_Rel = root.child_rel
                self.edu_stack.pop()
                self.edu_stack.pop()
                # 把前两个节点删除之后，把新的节点加入到stack，直接存入当前root节点即可
                self.edu_stack.append(root)
                temp_transition = "Reduce-" + node_NS + "-" + node_Rel
            print("当前操作 : ", temp_transition)
            # 对当前数据和操作标签进行存储
            self.train_label_list.append(temp_transition)
            self.train_data_list.append(self.feat)

    def get_features(self):
        queue_state = self.edu_queue[0] if len(self.edu_queue) > 0 else None
        stack_first_state = self.edu_stack[-1] if len(self.edu_stack) > 0 else None
        stack_second_state = self.edu_stack[-2] if len(self.edu_stack) > 1 else None
        self.feat = (stack_first_state, stack_second_state, queue_state)

    def init_queue_stack(self):
        """
        动态初始化，根据遍历到的文档初始化队列，栈设置为空
        树根的edu_node是从左到右的edu的顺序
        所以queue中只要append即可，虽然这样子不符合进队列的方式
        :return:
        """
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
        """
            build edu_list and edu_ids according to EDU files
            这里进行对关系类型转fine关系，再转coarse关系
        :param rst_dt_path:
        :param file_name:
        :return:
        """
        print(file_name)
        temp_path = os.path.join(rst_dt_path, file_name)
        edus_list, edu_span_list, edus_ids_list, edus_tag_ids_list, edus_conns_list, edu_headword_ids_list,\
            edu_has_center_word_list = get_edus_info(temp_path.replace(".dis", ".edus"),
                                                     temp_path.replace(".out.dis", ".out"),
                                                     nlp=self.nlp, file_name=file_name)
        dis_tree_obj = open(temp_path, 'r')
        lines_list = dis_tree_obj.readlines()
        # 加载关系转换映射表
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
        # root.pre_traverse()
        return root

    def build_tree_obj_list(self):
        """
        为生文本创建tree_obj对象
        :return:
        """
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
                    tmp_edu.temp_edu_heads = edu_headword_ids_list.pop(0)  # 当前EDU所有tokens对应的head word的ids
                    tmp_edu.temp_edu_has_center_word = edu_has_center_word_list.pop(0)  # 当前EDU是否包含句子中心词
                    tmp_edus_list.append(tmp_edu)
                # 构建tree_obj对象
                tmp_tree_obj = tree_obj()
                tmp_tree_obj.file_name = file_name
                tmp_tree_obj.assign_edus(tmp_edus_list)
                parse_trees.append(tmp_tree_obj)
        return parse_trees
