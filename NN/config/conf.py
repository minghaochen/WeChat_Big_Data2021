import os
import warnings
import pickle
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 存储数据的根目录

ROOT_PATH = os.path.join(BASE_DIR, '../data')
# 比赛数据集路径
DATASET_PATH = os.path.join(ROOT_PATH, 'wedata/wechat_algo_data1/')
TEST_B_PATH = os.path.join(ROOT_PATH, 'wedata/wechat_algo_data1_b/')
# 训练集
USER_ACTION = os.path.join(DATASET_PATH, "user_action.csv")
FEED_INFO = os.path.join(DATASET_PATH, "feed_info.csv")
FEED_EMBEDDINGS =  os.path.join(DATASET_PATH, "feed_embeddings.csv")
# 测试集
TEST_A_FILE = os.path.join(DATASET_PATH, "test_a.csv")
TEST_FILE = os.path.join(TEST_B_PATH, "test_b.csv")
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward", 'favorite', 'comment', 'follow']
# ACTION_LIST = ["read_comment"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id', 'description',
                 'manual_keyword_list', 'manual_tag_list']
# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 10, "comment": 10, "follow": 10,
                      "favorite": 10}

PRETRAIN_PATH = os.path.join(BASE_DIR, '../data/pretrain_model')

W2V_PATH = os.path.join(PRETRAIN_PATH, 'w2v_16')
if not os.path.exists(W2V_PATH): os.makedirs(W2V_PATH)

SUBMT_PATH = os.path.join(BASE_DIR, '../data/submission')

USER_DATA_PATH = os.path.join(BASE_DIR, '../data/user_data')

CHUNK_DATA_PATH = os.path.join(BASE_DIR, '../data/user_data/chunk_data')
