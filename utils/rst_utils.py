# -*- coding: utf-8 -*-

"""
@Author: lyzhan
@Date: 2018/4/5
@Description:  对论元对数据获取
"""
import nltk
from config import *
# from parser_model.parser_config import RST_TRAIN_LABEL, RST_LABEL_ALL_PKL, RST_LABEL_ALL_TSV, N_TAGS
import numpy as np
from utils.file_util import *
from utils.text_process_util import rm_edge_s, get_sent_words_syns, s_list
from nltk.tag import StanfordPOSTagger


def build_conn_dict():
    """
    根据RST语料库获取内容的词性，创建词性pos2ids字典.
    question: 一个词的词性是确定的还是根据单词所在的句子不同而改变的。
    :return:
    """
    corpus_vocab = load_data(VOC_WORD2IDS_PATH).keys()
    conn_word_set = set()
    conn_list = load_data(CONN_RAW_LIST)
    for conn in conn_list:
        tmp_words_set = set(conn.split())
        conn_word_set = conn_word_set | tmp_words_set
    conn_word_list = list(conn_word_set)
    conn_word_list.insert(0, UNK)
    conn_word_list.insert(0, PAD)
    conn2ids = dict()
    idx = 0
    for c_word in conn_word_list:
        if c_word in corpus_vocab:
            conn2ids[c_word] = idx
            idx += 1
    save_data(conn2ids, CONN_word2ids)
    write_iterate(list(conn2ids.keys()), CONN_WORD_File)


def build_pos_dict():
    """
    根据pdtb语料库获取内容的词性，创建词性pos2ids字典.
    question: 一个词的词性是确定的还是根据单词所在的句子不同而改变的。
    :return:
    """
    tag_set = set()
    corpus_dir_list = [RST_DT_TRAIN_PATH, RST_DT_TEST_PATH]
    for dir in corpus_dir_list:
        for filename in os.listdir(dir):
            if filename.endswith(".out"):
                file_ = os.path.join(dir, filename)
                with open(file_, "r") as f:
                    for line in f:
                        line = line.strip()
                        if len(line) == 0:
                            continue
                        tags = [tup[1] for tup in nltk.pos_tag(nltk.word_tokenize(line))]  # 这里没转小写，包含了符号
                        tag_set = tag_set | set(tags)
    # 最终得到当前语料库的tag集合
    tag_list = list(tag_set)
    tag_list.insert(0, UNK)
    tag_list.insert(0, PAD)
    pos2ids = dict()
    for tag in tag_list:
        pos2ids[tag] = tag_list.index(tag)
    save_data(pos2ids, POS_word2ids_PATH)
    write_iterate(list(pos2ids.keys()), POS_TAGS_PATH)


def build_glove_dict():
    """
    对指定维度的glove向量库创建数据字典
    :return:
    """
    dim = EMBED_SIZE
    save_dir = os.path.join(GLOVE_PATH, "gloves_" + str(dim))
    safe_mkdir(save_dir)
    file_tag = 0
    count_line = 0
    f_n = "part_"
    tmp_file = os.path.join(save_dir, f_n + str(file_tag) + ".pkl")
    tmp_dict = dict()
    word2file_dict = dict()
    with open(os.path.join(GLOVE_PATH, "glove.6B." + str(EMBED_SIZE) + "d.txt"), "r") as f:
        for line in f:
            count_line += 1
            line_ = line.strip().split()
            if len(line_) < 25:
                continue
            vec = np.array(line_[-dim:])
            word = ' '.join(line_[:-dim])
            tmp_dict[word] = vec.astype(np.float32)
            word2file_dict[word] = f_n + str(file_tag) + ".pkl"
            if count_line % 500000 == 0:  # 预计50万没问题
                with open(tmp_file, "wb") as f_:
                    pkl.dump(tmp_dict, f_)
                tmp_dict = dict()
                file_tag += 1
                tmp_file = os.path.join(save_dir, f_n + str(file_tag) + ".pkl")
        # 余数存储
        if len(tmp_dict.keys()) > 0:
            with open(tmp_file, "wb") as f_:
                pkl.dump(tmp_dict, f_)

    # 对文件名映射信息进行存储
    with open(os.path.join(save_dir, "word2filename.pkl"), "wb") as f:
        pkl.dump(word2file_dict, f)


def build_Siamese_word_dict():
    """
        对Siamese词向量的word dict创建, 同事创建针对这个向量库的训练数据和测试数据的EDU_ids
    """
    siamese_word_emb = load_data(Siamese_vec_PATH)  # VOC_VEC_Siamese_PATH
    train_edus_token_list = build_rst_edus_tokens("train")
    test_edus_token_list = build_rst_edus_tokens("test")
    edus_token_list = train_edus_token_list.copy()
    edus_token_list.extend(test_edus_token_list.copy())
    # 词频统计
    freq_count = dict()
    # 词库构建
    word2ids = dict()
    # 对<UNK>和0部分进行添加
    word2ids[PAD] = 0
    ids2vec = [np.zeros(shape=(EMBED_SIZE_Siamese,), dtype=np.float32)]
    word2ids[UNK] = 1
    ids2vec = np.append(ids2vec, [np.random.uniform(-0.25, 0.25, EMBED_SIZE_Siamese)], axis=0)  # 随机初始化
    idx = 2
    for edu_tokens in edus_token_list:
        for token in edu_tokens:
            # word = rm_edge_s(word)
            token = token.lower()
            if token in siamese_word_emb.keys():
                if token not in word2ids.keys():
                    word2ids[token] = idx
                    ids2vec = np.append(ids2vec, [siamese_word_emb[token]], axis=0)
                    freq_count[token] = 1
                    idx += 1
                else:
                    freq_count[token] += 1
            else:
                continue
    print("基本word2ids创建完毕！")
    # 低频词过滤
    for key in freq_count.keys():
        if freq_count[key] <= LOW_FREQ:
            del word2ids[key]

    # 重建
    word2ids_freqed = dict()
    ids2vec_freqed = []
    for idx, word in enumerate(word2ids.keys()):
        word2ids_freqed[word] = idx
        ids2vec_freqed.append(ids2vec[word2ids[word]])
    ids2vec_freqed = np.array(ids2vec_freqed, dtype=np.float32)
    print("低频词过滤完毕！")

    # 将获取的edus_list转换成eduids_list, 这里只是提供给CBOS训练使用
    train_ids_list = edus_toks2ids(train_edus_token_list, word2ids_freqed)
    test_ids_list = edus_toks2ids(test_edus_token_list, word2ids_freqed)
    print("edus转ids完毕！")
    # 存储
    save_data(word2ids_freqed, VOC_WORD2IDS_Siamese_PATH)
    save_data(ids2vec_freqed, VOC_VEC_Siamese_PATH)
    write_iterate(word2ids.keys(), WORDS_Siamese_PATH)
    save_data(train_ids_list, RST_TRAIN_EDUS_Siamese_IDS_PATH)
    save_data(test_ids_list, RST_TEST_EDUS_Siamese_IDS_PATH)


def build_glove_word_dict():
    """
        对词汇到文件名的映射，然后对向量的映射 创建word2ids和ids2vec
    """
    glove_path = os.path.join(GLOVE_PATH, "gloves_" + str(EMBED_SIZE))
    word2file_path = os.path.join(glove_path, "word2filename.pkl")
    word2file = load_data(word2file_path)
    word2ids_filename = dict()
    for filename in os.listdir(glove_path):
        if filename.startswith("part"):
            word2ids_filename[filename] = load_data(os.path.join(glove_path, filename))

    # 遍历edus文件并存储edus文本数据
    train_edus_token_list = build_rst_edus_tokens("train")
    test_edus_token_list = build_rst_edus_tokens("test")
    edus_token_list = train_edus_token_list.copy()
    edus_token_list.extend(test_edus_token_list.copy())
    print("构建ing")
    # 词频统计
    freq_count = dict()
    # 词库构建
    word2ids = dict()
    # 对<UNK>和0部分进行添加
    word2ids[PAD] = 0
    ids2vec = [np.zeros(shape=(EMBED_SIZE,), dtype=np.float32)]
    word2ids[UNK] = 1
    ids2vec = np.append(ids2vec, [np.random.uniform(-0.25, 0.25, EMBED_SIZE)], axis=0)  # 随机初始化
    idx = 2
    for edu_tokens in edus_token_list:
        for token in edu_tokens:
            # word = rm_edge_s(word)
            token = token.lower()
            if token in word2file.keys():
                if token not in word2ids.keys():
                    word2ids[token] = idx
                    ids2vec = np.append(ids2vec, [word2ids_filename[word2file[token]][token]], axis=0)
                    freq_count[token] = 1
                    idx += 1
                else:
                    freq_count[token] += 1
            else:
                continue
    print("基本word2ids创建完毕！")
    # 低频词过滤
    for key in freq_count.keys():
        if freq_count[key] <= LOW_FREQ:
            del word2ids[key]

    # 重建
    word2ids_freqed = dict()
    ids2vec_freqed = []
    for idx, word in enumerate(word2ids.keys()):
        word2ids_freqed[word] = idx
        ids2vec_freqed.append(ids2vec[word2ids[word]])
    ids2vec_freqed = np.array(ids2vec_freqed, dtype=np.float32)
    print("低频词过滤完毕！")

    # 将获取的edus_list转换成eduids_list
    train_ids_list = edus_toks2ids(train_edus_token_list, word2ids_freqed)
    test_ids_list = edus_toks2ids(test_edus_token_list, word2ids_freqed)
    print("edus转ids完毕！")
    # 存储
    save_data(word2ids_freqed, VOC_WORD2IDS_PATH)
    save_data(ids2vec_freqed, VOC_VEC_PATH)
    save_data(ids2vec_freqed, CBOS_VEC_PATH)
    write_iterate(word2ids.keys(), WORDS_PATH)
    save_data(train_ids_list, RST_TRAIN_EDUS_IDS_PATH)
    save_data(test_ids_list, RST_TEST_EDUS_IDS_PATH)


def build_rst_edus_tokens(type_):
    """
    :param type_: train or test
    :return:
    """
    edus_token_list = []
    if type_ is "train":
        txt_path_ = RST_DT_TRAIN_PATH
        edus_file_path = RST_TRAIN_EDUS_PATH
    elif type_ is "test":
        txt_path_ = RST_DT_TEST_PATH
        edus_file_path = RST_TEST_EDUS_PATH
    else:
        txt_path_ = edus_list_path = edus_file_path = None
        input("rst_utils  check!")

    for filename in os.listdir(txt_path_):
        if filename.endswith(".out.edus"):
            tmp_edu_file = os.path.join(txt_path_, filename)
            tmp_sent_file = tmp_edu_file.replace(".out.edus", ".out")
            tmp_edus_token_list = get_edus_tokens_list(tmp_edu_file, tmp_sent_file)
            edus_token_list.extend(tmp_edus_token_list)
    return edus_token_list


def edus_toks2ids(edu_toks_list, word2ids):
    """
    将训练cbos的论元句子们转换成ids序列， 将训练cdtb论元关系的论元对转成对应的论元对的tuple ids 列表并返回
    :return:
    """
    tok_list_ids = []
    for line in edu_toks_list:
        line_ids = get_line_ids(toks=line, word2ids=word2ids)
        tok_list_ids.append(line_ids)
    # 数据存储
    return tok_list_ids


def get_line_ids(toks, word2ids=None):
    """
    根据输入的行信息和word2ids字典对句子进行转换, no padding
    :param line:
    :param word2ids:
    :return:
    """
    if word2ids is None:
        input("rst_utils.py")
    line_ids = []
    for tok in toks:
        tok = tok.lower()
        if tok in word2ids.keys():
            line_ids.append(word2ids[tok])
        else:
            line_ids.append(word2ids[UNK])
    return line_ids


def judge_equal(sent_raw, edus_token_list):
    """
    判断字符级别相等, 若句子长度和edu联合长度相等返回1，若句子长度小于edu联合，返回2，否则返回0
    :param sent_raw:
    :param edus_token_list:
    :return:
    """
    flag = 0
    sent_raw = "".join(sent_raw.split())
    sent_temp = "".join(["".join(edus_tokens) for edus_tokens in edus_token_list])
    print("sent_raw: ", sent_raw)
    print("sent_tmp: ", sent_temp)
    print("checking...")
    if sent_raw == sent_temp:
        flag = 1
    if len(sent_raw) < len(sent_temp):
        flag = 2
    return flag


def extract_conn_ids(tmp_edu_tokens, conn2ids):
    tmp_edu_conn_ids = []
    for token in tmp_edu_tokens:
        if token in conn2ids.keys():
            tmp_edu_conn_ids.append(conn2ids[token])
    return tmp_edu_conn_ids


def convert2tokens_and_ids(edus_list, sent_list, word2ids):
    """
    # for tok in temp_edu_tokens:
                #     tok = tok.lower()
                #     tok_ids = word2ids[tok] if tok in word2ids.keys() else word2ids[UNK]
                #     temp_token_ids.append(tok_ids)
    :param edus_list:
    :param sent_list:
    :param word2ids:
    :return:
    """
    sents_token_list = []
    edus_token_list = []
    edus_token_ids_list = []

    temp_sent_ids = 0
    temp_edus_token_list = []
    temp_sents_token_list = None

    # 加载conn2ids
    conn2ids = load_data(CONN_word2ids)
    edus_conns_list = []
    # temp_sent_ 代表的是比较啊的当前句
    temp_sent_ = sent_list[temp_sent_ids]
    for edu in edus_list:
        tmp_edu_tokens = get_sent_words_syns(edu)
        edus_conns_list.append(extract_conn_ids(tmp_edu_tokens, conn2ids))
        temp_edus_token_list.append(tmp_edu_tokens)
        judge_out = judge_equal(temp_sent_, temp_edus_token_list)
        if judge_out == 0:
            continue
        elif judge_out == 2:
            temp_sent_ids += 1
            temp_sent_ = temp_sent_ + " " + sent_list[temp_sent_ids]
            judge_out = judge_equal(temp_sent_, temp_edus_token_list)

        if judge_out == 1:
            print("匹配一句成功.")
            temp_sent_ids += 1
            temp_sent_ = sent_list[temp_sent_ids] if temp_sent_ids < len(sent_list) else None
            # 下面对当前一组EDUs修改引号、组件句子、创建edu_ids信息
            flag = True
            for temp_edu_tokens in temp_edus_token_list:
                temp_token_ids = []
                for idx in range(len(temp_edu_tokens)):
                    if temp_edu_tokens[idx] == "\"":
                        temp_edu_tokens[idx] = "``" if flag else "''"
                        flag = not flag
                    tok = temp_edu_tokens[idx].lower()
                    tok_ids = word2ids[tok] if tok in word2ids.keys() else word2ids[UNK]
                    temp_token_ids.append(tok_ids)
                # 记录转换之后的edu级tokens和对应的ids
                edus_token_list.append(temp_edu_tokens.copy())
                edus_token_ids_list.append(temp_token_ids)
                # 用当前遍历的EDU 组成句子token
                if temp_sents_token_list is None:
                    temp_sents_token_list = temp_edu_tokens
                else:
                    temp_sents_token_list.extend(temp_edu_tokens)
            # 记录转换之后的句子级别的tokens
            sents_token_list.append(temp_sents_token_list)
            temp_sents_token_list = None
            temp_edus_token_list = []
    return edus_token_list, edus_token_ids_list, sents_token_list, edus_conns_list


def convert2tokens(edus_list, sent_list):
    temp_sent_ids = 0
    edus_token_list = []
    temp_edus_token_list = []
    for edu in edus_list:
        temp_edus_token_list.append(get_sent_words_syns(edu))
        if judge_equal(sent_list[temp_sent_ids], temp_edus_token_list):
            temp_sent_ids += 1
            flag = True
            for temp_edu_tokens in temp_edus_token_list:
                for idx in range(len(temp_edu_tokens)):
                    if temp_edu_tokens[idx] == "\"":
                        temp_edu_tokens[idx] = "``" if flag else "''"
                        flag = not flag
                edus_token_list.append(temp_edu_tokens)
            temp_edus_token_list = []
    return edus_token_list


def get_edus_tokens_list(edu_path, sentence_path):
    edus_list = []
    sent_list = []
    # 创建好edu_list和sent_list
    with open(sentence_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) is 0:
                continue
            sent_list.append(line)

    with open(edu_path, "r") as f:
        for line in f:
            edus_list.append(line.strip())

    edus_token_list = convert2tokens(edus_list, sent_list)
    return edus_token_list


def get_edus_info(edu_path, sentence_path, nlp=None, file_name=None):
    """
    获取EDU信息 对句子级别的词性信息获取并分配到EDU信息中进行返回，返回每个EDU的对应pos ids返回值
    生成每个EDU的的head word对应的ids 记录当前句子中的中心词是否在当前EDU
    nlp.dependency_parse(sent)
    :param edu_path:
    :return:
    """
    if not USE_Siamese:
        word2ids = load_data(VOC_WORD2IDS_PATH)
    else:
        word2ids = load_data(VOC_WORD2IDS_Siamese_PATH)
    pos2ids = load_data(POS_word2ids_PATH)
    edus_tag_ids_list = []
    edu_span_list = []
    edu_headword_ids_list = []
    edu_has_center_word_list = []

    # 创建好edu_list和sent_list
    sent_list = []
    with open(sentence_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) is 0:
                continue
            sent_list.append(line)

    edus_list = []
    with open(edu_path, "r") as f:
        for line in f:
            temp_edu = line.strip()
            edus_list.append(temp_edu)
    # 转换
    edus_token_list, edus_token_ids_list, sents_token_list, edus_conns_list \
        = convert2tokens_and_ids(edus_list, sent_list, word2ids)
    print("length of edu token list: ", len(edus_token_list))
    print("转换完成：edus-->tok_ids")

    # 对所有的句子词性获取, 依存信息获取
    tag_id_list = []
    dependency_ids_list = []
    for sents_token in sents_token_list:
        """
            以句子为单位进行句法分析和词性标注
        """
        # 词性标注
        tags = [tup[1] for tup in nltk.pos_tag(sents_token)]
        for tag in tags:
            tmp_tag_id = pos2ids[tag] if tag in pos2ids.keys() else pos2ids[UNK]
            tag_id_list.append(tmp_tag_id)

        # 句法分析
        new_sent = " ".join(sents_token)
        # 生成依存元祖，包含多棵树
        dependency_tuple_list = nlp.dependency_parse(new_sent)
        print(len(sents_token))
        print(len(dependency_tuple_list))

        if len(sents_token) != len(dependency_tuple_list):
            print(sents_token)
            print(new_sent)
            input(dependency_tuple_list)
        # 存储当前树的依存情况 tmp_word_index 2 word_ids of tmp_word's head word
        tmp_dependency_dict = dict()
        # 计数器，当前树的元祖个数
        count_tuple = 0
        for tuple_ in dependency_tuple_list:
            # 遇到新的树并且不是最开始, 将之前存储的树元素的信息存储
            if tuple_[1] == 0 and count_tuple > 0:
                for idx in range(1, count_tuple+1):  # 树的元素下标从1开始
                    dependency_ids_list.append(tmp_dependency_dict[idx])
                count_tuple = 0
                tmp_dependency_dict = dict()
                word_ids = None
                print("tree over1 ! ", file_name)
            else:
                word = sents_token[tuple_[1]-1]
                print(word)
                word_ids = word2ids[word] if word in word2ids.keys() else word2ids[UNK]
            print(tuple_)
            tmp_dependency_dict[tuple_[2]] = word_ids  # 没有head_word的就是中心词
            count_tuple += 1
        # 对一个句子中最后一棵句法依存树的信息存储
        for idx in range(1, count_tuple + 1):
            dependency_ids_list.append(tmp_dependency_dict[idx])
        print("===============tree over======================", file_name)

    # tag信息，dependency信息装载
    print(len(tag_id_list))
    print(len(dependency_ids_list))
    offset = 0
    for one_edu_tokens in edus_token_list:
        edu_span_list.append((offset, offset + len(one_edu_tokens) - 1))  # edu的起止跨度
        edu_has_center_word_list.append(False)  # 当前EDU是否包含句子中心词的标志
        # 词性标签制作
        temp_tags_ids = []
        temp_dep_idx = []
        for _ in one_edu_tokens:
            temp_tags_ids.append(tag_id_list.pop(0))
            dep_word_idx = dependency_ids_list.pop(0)
            if dep_word_idx is None:
                # 是中心词，做两个工作 当前EDU包含center word ，当前词汇中心词的head word设置为pad
                edu_has_center_word_list[-1] = True
                temp_dep_idx.append(PAD_ids)
            else:
                # 其它词汇
                temp_dep_idx.append(dep_word_idx)
        edus_tag_ids_list.append(temp_tags_ids)
        edu_headword_ids_list.append(temp_dep_idx)
        offset += len(one_edu_tokens)
    print("完成对每个EDU句子的tag_ids填充")
    return edus_list, edu_span_list, edus_token_ids_list, edus_tag_ids_list, edus_conns_list, edu_headword_ids_list, edu_has_center_word_list

# def convert_label2ids_fine(label_txt, convert2fine=None):
#     """
#     对label文本转换到细粒度向量 one-hot
#     :param label_txt:
#     :return:
#     """
#     if convert2fine is None:
#         convert2fine = load_data(CONV2Fine)
#     label_txt = convert2fine[label_txt]
#     with open(RST_FINE_REL, "rb") as f:
#         labels_list = pkl.load(f)
#     label_ids = np.zeros(shape=(FINE_REL_NUM,))
#     label_ids[labels_list.index(label_txt)] = 1
#     return label_ids
#
# def convert_label2idx_fine(label_txt, convert2fine=None):
#     """
#     对label文本转换到细粒度下标
#     :param label_txt:
#     :return:
#     """
#     if convert2fine is None:
#         convert2fine = load_data(CONV2Fine)
#     label_txt = convert2fine[label_txt]
#     with open(RST_FINE_REL, "rb") as f:
#         labels_list = pkl.load(f)
#     label_idx = labels_list.index(label_txt)
#     return label_idx
#
# def convert_label2ids_coarse(label_txt, convert2fine=None, convert2coarse=None):
#     """
#     对label文本转换粗粒度向量 one-hot
#     :param label_txt:
#     :return:
#     """
#     if convert2fine is None:
#         convert2fine = load_data(CONV2Fine)
#     label_txt = convert2fine[label_txt]
#     if convert2coarse is None:
#         convert2coarse = load_data(CONV2Coarse)
#     label_txt = convert2coarse[label_txt]
#     with open(RST_COARSE_REL, "rb") as f:
#         labels_list = pkl.load(f)
#     label_ids = np.zeros(shape=(COARSE_REL_NUM,))
#     label_ids[labels_list.index(label_txt)] = 1
#     return label_ids
#
# def convert_label2idx_coarse(label_txt, convert2fine=None, convert2coarse=None):
#     """
#     对label文本转换到粗粒度下标
#     :param label_txt:
#     :return:
#     """
#     if convert2fine is None:
#         convert2fine = load_data(CONV2Fine)
#     label_txt = convert2fine[label_txt]
#     if convert2coarse is None:
#         convert2coarse = load_data(CONV2Coarse)
#     label_txt = convert2coarse[label_txt]
#     with open(RST_COARSE_REL, "rb") as f:
#         labels_list = pkl.load(f)
#     label_idx = labels_list.index(label_txt)
#     return label_idx
