# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description: Global Configuration
"""
from utils.file_util import load_data

# Data.glove
GLOVE_PATH = "data/gloves"
GLOVE50_embedding_dir = "data/gloves/gloves_50"
GLOVE50_vec_dir = "data/gloves_50/embed.pkl"
GLOVE100_embedding_dir = "data/gloves/gloves_100"
GLOVE100_vec_dir = "data/gloves_100/embed.pkl"

# Siamese vec 外部引入
Siamese_vec_PATH = "data/Siamese_emb/emb4edu.pkl"
WORDS_Siamese_PATH = "data/voc_siamese/words.tsv"
VOC_WORD2IDS_Siamese_PATH = "data/voc_siamese/word2ids.pkl"
VOC_VEC_Siamese_PATH = "data/voc_siamese/ids2vec.pkl"

# Siamese vec rst 学习得到的
Siamese_CBOW_EMBED_PATH = "data/voc/cbow_embed.pkl"
Siamese_CBOW_VEC_PATH = "data/voc/cbow_vec.pkl"

# Data.voc & pos & so on 针对当前语料库的词库形成
WORDS_PATH = "data/voc/words.tsv"
VOC_WORD2IDS_PATH = "data/voc/word2ids.pkl"
VOC_VEC_PATH = "data/voc/ids2vec.pkl"
CBOS_EMBED_PATH = "data/voc/cbos_embed.pkl"
CBOS_VEC_PATH = "data/voc/cbos_vec.pkl"
POS_word2ids_PATH = "data/voc/pos2ids.pkl"
POS_TAGS_PATH = "data/voc/pos_tags.tsv"
CONN_RAW_LIST = "data/connective/conn_pdtb_list.pkl"
CONN_word2ids = "data/connective/conn2ids.pkl"
CONN_WORD_File = "data/connective/conn_word.tsv"

# Data.RST
RST_DT_TRAIN_PATH = "data/rst_dt/TRAINING"
RST_DT_TEST_PATH = "data/rst_dt/TEST"

RST_TRAIN_EDUS_PATH = "data/rst_dt/RST_EDUS/train_edus.tsv"
RST_TEST_EDUS_PATH = "data/rst_dt/RST_EDUS/test_edus.tsv"
RST_TRAIN_EDUS_IDS_PATH = "data/rst_dt/RST_EDUS/train_edus_ids.pkl"
RST_TRAIN_EDUS_Siamese_IDS_PATH = "data/rst_dt/RST_EDUS/train_edus_ids_sia.pkl"
RST_TEST_EDUS_IDS_PATH = "data/rst_dt/RST_EDUS/test_edus_ids.pkl"
RST_TEST_EDUS_Siamese_IDS_PATH = "data/rst_dt/RST_EDUS/test_edus_ids_sia.pkl"

# RST_FINE_REL = "data/rst_dt/RST_REL/fine_rel.pkl"
# RST_COARSE_REL = "data/rst_dt/RST_REL/coarse_rel.pkl"
REL_raw2coarse = "data/rst_dt/RST_REL/rel_raw2coarse.pkl"
REL_coarse2ids = "data/rst_dt/RST_REL/coarse2ids.pkl"
REL_ids2coarse = "data/rst_dt/RST_REL/ids2coarse.pkl"

RST_TRAIN_TREES = "data/rst_dt/train_trees.pkl"
RST_TRAIN_Siamese_TREES = "data/rst_dt/train_trees_sia.pkl"
RST_TEST_TREES = "data/rst_dt/test_trees.pkl"
RST_TEST_Siamese_TREES = "data/rst_dt/test_trees_sia.pkl"

LOG_DIR = "data/log_file"

PRETRAINED = "data/rst_dt/models_saved/parsing_model/rel_max_model.pth"
RAW_TXT = "data/raw_txt"
TREES_PARSED = "data/trees_parsed"

# Config.global
# 2: 2+1, 3: cnn, 4: 3+1 or bow, 5: self att + bilstm, 6: features embedding 7: 5 + 6   8: 3 + 6
VERSION = 5
SET_of_version = "347"
USE_FEATURE = False
USE_POS = True
USE_bow = False
USE_tracking = True
USE_conn_tracker = True
Use_Bi_cnn = False
USE_Siamese = False
USE_beam_search = False
DESC = "~ 两个tracker同时使用，256维度，cu04 data, 最新评测, BiLSTM编码器，词向量静默状态 ~"

Only_structure = True
REL_CLS = True
NO_STRUCTURE_LEARN = False  # 只考虑span不考虑NS关系
word_emb_learn = False
USE_dependency = False
IS_Coarse = True
EMBED_SIZE = 100
POS_EMBED_SIZE = 50  # 46个
FEAT2_SIZE = 6
FEAT3_SIZE = 3
FEAT4_SIZE = 2
FEAT5_SIZE = 2
FEAT6_SIZE = 5  # 18个关系类别 + 1个不存在关系的状态
FEATs_SIZE = 328

CONN_EMBED_SIZE = 100  # 100 -> 64 -> 100
FEATURE_EMBED_SIZE = 32  # 学到所有特征的embedding，每个特征组合对应一个向量 24个
PAD_SIZE = 50
UNK = "<UNK>"
UNK_ids = 1
PAD = "<PAD>"
PAD_ids = 0
LOW_FREQ = 1
FINE_REL_NUM = 56
COARSE_REL_NUM = 18
if not Only_structure:
    Transition_num = 43  # 结构关系 4 粗粒度形态43个
else:
    Transition_num = 4

proj_dropout = 0.2
l2_penalty = 1e-5
BEAM_SIZE = int(Transition_num / 2)
# 操作标签 只关心核型
SHIFT = "SHIFT"
REDUCE = "REDUCE"
REDUCE_NN = "REDUCE-NN"
REDUCE_NS = "REDUCE-NS"
REDUCE_SN = "REDUCE-SN"

# 操作标签 关心粗粒度关系
Action2ids_path = "data/rst_dt/action2ids.pkl"
Ids2action_path = "data/rst_dt/ids2action.pkl"

if not Only_structure:
    action2ids = load_data(Action2ids_path)
    ids2action = load_data(Ids2action_path)
else:
    action2ids = {SHIFT: 0, REDUCE_NN: 1, REDUCE_NS: 2, REDUCE_SN: 3}
    ids2action = {0: SHIFT, 1: REDUCE_NN, 2: REDUCE_NS, 3: REDUCE_SN}

coarse2ids = load_data(REL_coarse2ids)
ids2coarse = load_data(REL_ids2coarse)

# EDU特征标签
edu_feature2ids = {'T_l1_p1': 0, 'T_l1_p2': 1, 'T_l1_p3': 2, 'T_l1_p4': 3, 'T_l2_p1': 4, 'T_l2_p2': 5, 'T_l2_p3': 6,
                   'T_l2_p4': 7, 'T_l3_p1': 8, 'T_l3_p2': 9, 'T_l3_p3': 10, 'T_l3_p4': 11, 'F_l1_p1': 12, 'F_l1_p2': 13,
                   'F_l1_p3': 14, 'F_l1_p4': 15, 'F_l2_p1': 16, 'F_l2_p2': 17, 'F_l2_p3': 18, 'F_l2_p4': 19,
                   'F_l3_p1': 20, 'F_l3_p2': 21, 'F_l3_p3': 22, 'F_l3_p4': 23, 'PAD': 24}

ids2edu_feature = {0: 'T_l1_p1', 1: 'T_l1_p2', 2: 'T_l1_p3', 3: 'T_l1_p4', 4: 'T_l2_p1', 5: 'T_l2_p2', 6: 'T_l2_p3',
                   7: 'T_l2_p4', 8: 'T_l3_p1', 9: 'T_l3_p2', 10: 'T_l3_p3', 11: 'T_l3_p4', 12: 'F_l1_p1', 13: 'F_l1_p2',
                   14: 'F_l1_p3', 15: 'F_l1_p4', 16: 'F_l2_p1', 17: 'F_l2_p2', 18: 'F_l2_p3', 19: 'F_l2_p4',
                   20: 'F_l3_p1', 21: 'F_l3_p2', 22: 'F_l3_p3', 23: 'F_l3_p4', 24: 'PAD'}
HEAD_pad_ids = 24

# Config.Segmentation
seg_model_path = "data/segmentation_model_pkl/model.pickle.gz"
seg_voc_path = "data/segmentation_model_pkl/vocab.pickle.gz"

# Config.CBOS & CBOW
BATCH_SIZE_cbos = 500
SKIP_WINDOW = 2  # context上下文的单向跨度
NUM_SAMPLED = 2  # 负采样个数
LEARNING_RATE_cbos = 0.01  # 学习率
NUM_TRAIN_STEPS_cbos = 2000000
SKIP_STEP = 500  # 每5000次输出一次训练情况 打印loss等
VISUAL_PATH = 'visualization'  # 对句子表示的展示
NUM_VISUALIZE = 1000  # number of edus to visualize
LOSS_PATH = "data/Loss_data/"
# DEV_LOSS_PATH = "data/Loss_data/dev_loss.pkl"

# Config.spinn
SEED = 2
SPINN_HIDDEN = 128
SPINN_MODEL = "data/rst_dt/spinn_model"
NUM_TRAIN_STEPS_spinn = 1000
LEARNING_RATE_spinn = 0.001
BATCH_SIZE_spinn = 1
SKIP_STEP_spinn = 6

# spinn.cnn
# cnn parameters
IN_CHANNEL_NUM = 1
FILTER_NUM = 128
FILTER_ROW = 1  # Bi_gram
if USE_POS:
    FILTER_COL = EMBED_SIZE + POS_EMBED_SIZE  # EMBEDDING_SIZE
else:
    FILTER_COL = EMBED_SIZE
STRIDE = 1
# pooling 窗口根据convolution的结果设定
KERNEL_ROW = PAD_SIZE
KERNEL_COL = 1

# mlp parameters (1, 584)  328 + 256 = 584
if USE_FEATURE:
    mlp_input_size = SPINN_HIDDEN * 2 + FEATs_SIZE if USE_conn_tracker else SPINN_HIDDEN + FEATs_SIZE
else:
    mlp_input_size = SPINN_HIDDEN * 2 if USE_conn_tracker else SPINN_HIDDEN
mlp_hidden_size = int(mlp_input_size / 2)  # 隐藏层单元数设置为输入层单元数的一半

# mlp_hidden_size = SPINN_HIDDEN
# if not USE_FEATURE:
#     if USE_conn_tracker:
#         mlp_input_size = 2 * mlp_hidden_size
#     else:
#         mlp_input_size = mlp_hidden_size
# else:
#     if USE_conn_tracker:
#         mlp_input_size = 2 * mlp_hidden_size + FEATs_SIZE
#     else:
#         mlp_input_size = mlp_hidden_size + FEATs_SIZE
mlp_num_layers = 2  # 减1
mlp_dropout = 0.2

# model_saved path
MODELS2SAVE = "data/rst_dt/models_saved"
LOAD_TEST = False
