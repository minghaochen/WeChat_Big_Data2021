# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os
import psutil
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../model'))
from mmoe import *
sys.path.append(os.path.join(BASE_DIR, '../third_party'))
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import random
sys.path.append(os.path.join(BASE_DIR, '..'))
from evaluation_fast import uAUC
import time as ti
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from conf import *

def seed_everything(seed):   # seed
    """
    Seeds basic parameters for reproductibility of results
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 6666
seed_everything(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEBUG_MODE = False
USE_DR = True
FEED_DIM = 20
FEED_DIM_ = 16
EPOCHS = 3
BATCH = 1024
cycle_epoch = 3

model_save_dir = os.path.join(ROOT_PATH, 'model/mmoe_checkpoints/')
submit_dir = os.path.join(ROOT_PATH, "single_model_result")
if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)

def get_dense_feature():
    # 提取连续特征
    con_features = []

    con_features.append('feed_count_current_day')

    con_features.append('user_count_current_day')

    return con_features

def get_other_dense_features():

    con_features = []

    prefix = 'user_feed_15_day_'
    con_features += [prefix+str(i) for i in range(FEED_DIM_)]
    #
    prefix = 'multi_mode_feed_'
    con_features += [prefix+str(i) for i in range(FEED_DIM_)]

    prefix = 'feed_user_15_day_'
    con_features += [prefix+str(i) for i in range(FEED_DIM_)]

    prefix = 'user_feed_tfidf_'
    con_features += [prefix+str(i) for i in range(64)]

    prefix = 'feed_user_tfidf_'
    con_features += [prefix+str(i) for i in range(64)]

    return con_features

# def split(x):
#     key_ans = x.split(dem)
#     for key in key_ans:
#         if key not in key2index:
#             # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
#             key2index[key] = len(key2index) + 1
#     return list(map(lambda x: key2index[x], key_ans))
#
# def split_v2(x):
#     key_ans = x
#     for key in key_ans:
#         if key not in key2index:
#             # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
#             key2index[key] = len(key2index) + 1
#     return list(map(lambda x: key2index[x], key_ans))


if __name__ == "__main__":
    print("*" * 20 + "MMOE模块" + "*" * 20)
    t1 = ti.time()
    submit = pd.read_csv(TEST_FILE)[['userid', 'feedid']]
    target = ["read_comment", "like", "click_avatar", "forward", 'favorite', 'comment', 'follow']
    train_all = pd.read_csv(os.path.join(ROOT_PATH, "train_all_data_chusai.csv"))
    valid_days = [14]
    seed_list = [1024]
    test_preds = []

    test = pd.read_csv(os.path.join(ROOT_PATH, "test_data_chusai_b.csv"))
    test['date_'] = 15
    sparse_features = ['new_user_id', 'new_feedid', 'new_authorid', 'new_bgm_song_id', 'new_bgm_singer_id', 'device']
    dense_features = ['videoplayseconds']
    history_feature_list = ['new_feedid']
    USE_FEAT = sparse_features + ['date_']
    test = test[USE_FEAT]
    tra = train_all
    actions_of_train = tra[target].reset_index(drop=True)
    tra = tra[USE_FEAT]
    train_len = tra.shape[0]
    all_data = pd.concat((tra, test)).reset_index(drop=True)

    # 特征
    local_exfeatures_con = get_dense_feature()
    dense_features += local_exfeatures_con # 连续特征

    # varlen
    varlen_feature_columns = []
    varlen_feature_columns += [VarLenSparseFeat(
        SparseFeat('description', vocabulary_size=150860, embedding_dim=FEED_DIM), maxlen=256, combiner='sum')]

    varlen_feature_columns += [VarLenSparseFeat(SparseFeat('manual_tag_list', vocabulary_size=352,
                                                           embedding_dim=FEED_DIM), maxlen=4, combiner='sum')]

    varlen_feature_columns += [VarLenSparseFeat(SparseFeat('manual_keyword_list', vocabulary_size=25743,
                                                           embedding_dim=FEED_DIM), maxlen=4, combiner='sum')]

    fixlen_feature_columns = []
    fixlen_feature_columns += [SparseFeat('new_user_id', all_data['new_user_id'].max()+1, embedding_dim=FEED_DIM)]
    fixlen_feature_columns += [SparseFeat('new_feedid', all_data['new_feedid'].max() + 2, embedding_dim=FEED_DIM)]
    fixlen_feature_columns += [SparseFeat('new_authorid', all_data['new_authorid'].max() + 1, embedding_dim=FEED_DIM)]
    fixlen_feature_columns += [SparseFeat('new_bgm_song_id', all_data['new_bgm_song_id'].max() + 1, embedding_dim=FEED_DIM)]
    fixlen_feature_columns += [SparseFeat('new_bgm_singer_id', all_data['new_bgm_singer_id'].max() + 1, embedding_dim=FEED_DIM)]
    fixlen_feature_columns += [SparseFeat('device', all_data['device'].max() + 1, embedding_dim=FEED_DIM)]

    keep_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(keep_feature_columns)

    dense_features += get_other_dense_features()
    fixlen_feature_columns += [DenseFeat(feat, 1, ) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns

    print(f"fixlen_feature_columns：{len(fixlen_feature_columns)}")
    print(f"varlen_feature_columns：{len(varlen_feature_columns)}")

    # 3.generate input data for model
    tra, test = all_data.iloc[:train_len], all_data.iloc[train_len:]
    tra = pd.concat((tra, actions_of_train), axis=1)

    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    t2 = ti.time()
    ts = (t2 - t1) / 3600.0
    print(f"prepare data take time:{ts:.3f}")

    for i, d in enumerate(valid_days):
        print(f"valid day {d}")
        train, train_index = tra[~(tra['date_']==d)], tra[~(tra['date_']==d)].index.tolist()
        valid, valid_index = tra[tra['date_']==d], tra[tra['date_']==d].index.tolist()
        test_index = test.index.tolist()

        userid_list = valid['new_user_id'].astype(str).tolist()
        train_model_input = {name: train[name] for name in feature_names}
        valid_model_input = {name: valid[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}

        train_labels = [train[y].values for y in target]
        train_labels = np.concatenate([np.expand_dims(i, axis=1) for i in train_labels], axis=1)
        val_labels = [valid[y].values for y in target]
        val_labels = np.concatenate([np.expand_dims(i, axis=1) for i in val_labels], axis=1)

        weight_auc = 0.
        for seed in seed_list:
            model = MMoE_V5(
                linear_feature_columns=linear_feature_columns,
                dnn_feature_columns=dnn_feature_columns,
                dnn_hidden_units=(512, 256),
                task_dnn_units=None,
                num_experts=8, num_tasks=len(target), expert_dim=16,
                tasks=['binary']*len(target),
                device=device,
                seed=seed,
            )
            train_len = round(train.shape[0] / BATCH)
            model.compile(optimizer='adagrad', snap=False, loss="binary_crossentropy",
                          metrics=["binary_crossentropy", "auc"], lr=1e-2)

            info = psutil.virtual_memory()
            print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
            print(u'当前使用的总内存占比：', info.percent)

            for epoch in range(1, EPOCHS+1):
                print(f"{epoch}/{EPOCHS}")
                history = model.fit(train_model_input, train_labels, extra_data=train['date_'].values, batch_size=BATCH, epochs=1, verbose=1, current_epoch=epoch)
                val_pred_ans = model.predict(valid_model_input, extra_data=valid['date_'].values, batch_size=BATCH*2)
                val_uauc, _ = uAUC(val_labels, val_pred_ans, userid_list, target)
                if epoch % cycle_epoch == 0:
                    print(f"cycle: {epoch/cycle_epoch}/{EPOCHS/cycle_epoch}")
                    model_save_path = os.path.join(model_save_dir, f'mmoe_model_fold{i+1}.pth')
                    print(model_save_path)
                    torch.save(model.state_dict(), model_save_path)
                    # save dnn weight
                    t1 = ti.time()
                    pred_ans = model.predict(test_model_input, extra_data=test['date_'].values, batch_size=BATCH*5)
                    t2 = ti.time()
                    print(f'{len(target)}个目标行为%d条样本预测耗时（毫秒）：%.3f' % (len(test), (t2 - t1) * 1000.0))
                    ts = (t2 - t1) * 1000.0 / len(test) * 2000.0
                    print(f'{len(target)}个目标行为2000条样本平均预测耗时（毫秒）：%.3f' % ts)
                    test_preds.append(pred_ans)
            torch.cuda.empty_cache()

    # 5.生成提交文件
    for i, action in enumerate(target):
        submit[action] = 0.
        for f in test_preds:
            submit[action] += f[:, i].squeeze()
    submit[['userid', 'feedid'] + target].to_csv(os.path.join(submit_dir, 'mmoe.csv'), index=None, float_format='%.6f')