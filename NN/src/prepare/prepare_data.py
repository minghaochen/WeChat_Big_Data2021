# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from conf import *
# print(TEST_FILE)
hist_seq_feature = False

def process_embed_to_array(train:pd.DataFrame):
    feed_embed_array = np.zeros((train.shape[0], 513))
    for i in tqdm(range(train.shape[0])):
        x = train.loc[i, 'feed_embedding']
        y = [train.loc[i, 'feedid']] + [float(i) for i in str(x).strip().split(" ")]
        feed_embed_array[i, :] = y
    np.save(os.path.join(ROOT_PATH, 'feed_embedding.npy'), feed_embed_array)

def prepare_data():
    print("*" * 20 + "数据预处理" + "*" * 20)
    global  FEA_FEED_LIST
    feed_info_df = pd.read_csv(FEED_INFO)
    feed_info_df = feed_info_df.drop_duplicates(['feedid'], keep='first')
    feed_info_df = feed_info_df.sort_values(by=['feedid'], ascending=True)
    feed_info_df['new_feedid'] = list(range(feed_info_df.shape[0]))

    feed_info_df["bgm_song_id"] = feed_info_df["bgm_song_id"].fillna(feed_info_df["bgm_song_id"].max()+1)
    feed_info_df["bgm_singer_id"] = feed_info_df["bgm_singer_id"].fillna(feed_info_df["bgm_singer_id"].max()+1)
    feed_info_df["videoplayseconds"] = feed_info_df["videoplayseconds"].fillna(0.)


    # fix manual tag/key list
    # for i in range(feed_info_df.shape[0]):
    #     f1, f2 = feed_info_df.loc[i, 'manual_tag_list'], feed_info_df.loc[i, 'manual_keyword_list']
    #     f1, f2 = sorted([int(i) for i in f1.split(';')]), sorted([int(i) for i in f2.split(';')])
    #     f1, f2 = [str(i) for i in f1], [str(i) for i in f2]
    #     f1 = ";".join(f1)
    #     f2 = ";".join(f2)
    #     feed_info_df.loc[i, 'manual_tag_list'] = f1
    #     feed_info_df.loc[i, 'manual_keyword_list'] = f2

    # # convert machine tag
    # machine_tag_list = []
    # for i in range(feed_info_df.shape[0]):
    #     f = feed_info_df.loc[i, 'machine_tag_list']
    #     if f == '-1':
    #         machine_tag_list.append(f)
    #         continue
    #     temp = f.split(";")
    #     tag = []
    #     for j in temp:
    #         if float(j.split(" ")[1]) > 0.1:
    #             tag.append(j.split(" ")[0])
    #     machine_tag_list.append(";".join(tag))
    # feed_info_df['machine_tag_list_fix'] = machine_tag_list
    # FEA_FEED_LIST += ['machine_tag_list_fix']

    authorid_df = feed_info_df.drop_duplicates(['authorid'], keep='first')[['authorid']]
    authorid_df = authorid_df.sort_values(by=['authorid'], ascending=True)
    authorid_df['new_authorid'] = list(range(authorid_df.shape[0]))
    feed_info_df = pd.merge(feed_info_df, authorid_df, on=['authorid'], how='left')

    bgm_song_id_df = feed_info_df.drop_duplicates(['bgm_song_id'], keep='first')[['bgm_song_id']]
    bgm_song_id_df = bgm_song_id_df.sort_values(by=['bgm_song_id'], ascending=True)
    bgm_song_id_df['new_bgm_song_id'] = list(range(bgm_song_id_df.shape[0]))
    feed_info_df = pd.merge(feed_info_df, bgm_song_id_df, on=['bgm_song_id'], how='left')

    bgm_singer_id_df = feed_info_df.drop_duplicates(['bgm_singer_id'], keep='first')[['bgm_singer_id']]
    bgm_singer_id_df = bgm_singer_id_df.sort_values(by=['bgm_singer_id'], ascending=True)
    bgm_singer_id_df['new_bgm_singer_id'] = list(range(bgm_singer_id_df.shape[0]))
    feed_info_df = pd.merge(feed_info_df, bgm_singer_id_df, on=['bgm_singer_id'], how='left')

    feed_info_df['bgm_song_id'] = feed_info_df['bgm_song_id'].astype('int64')
    feed_info_df['bgm_singer_id'] = feed_info_df['bgm_singer_id'].astype('int64')

    user_action_df = pd.read_csv(USER_ACTION)
    user_id_df = user_action_df.drop_duplicates(['userid'], keep='first')[['userid']]
    user_id_df = user_id_df.sort_values(by=['userid'], ascending=True)
    user_id_df['new_user_id'] = list(range(user_id_df.shape[0]))
    user_action_df = pd.merge(user_action_df, user_id_df, on=['userid'], how='left')

    feed_embed = pd.read_csv(FEED_EMBEDDINGS)
    feed_embed = feed_embed.drop_duplicates(['feedid'], keep='first')
    feed_embed = feed_embed.sort_values(by=['feedid'], ascending=True)
    process_embed_to_array(feed_embed)

    test = pd.read_csv(TEST_FILE)
    test['date_'] = 15
    # add feed feature
    train = pd.merge(user_action_df, feed_info_df[FEA_FEED_LIST+['new_feedid', 'new_authorid', 'new_bgm_song_id',
                                                                 'new_bgm_singer_id']], on='feedid', how='left')

    test = pd.merge(test, user_id_df, on=['userid'], how='left')
    test = pd.merge(test, feed_info_df[FEA_FEED_LIST+['new_feedid', 'new_authorid', 'new_bgm_song_id',
                                                                 'new_bgm_singer_id']], on='feedid', how='left')
    COLUMNS = ['userid', 'feedid', 'date_', 'authorid']
    user_action = train[COLUMNS]
    test_user_action = test[COLUMNS]
    all_behavior = user_action.append(test_user_action)

    if hist_seq_feature:
        temp = all_behavior.groupby(['new_user_id'])['new_feedid'].apply(list).reset_index()
        temp = temp.rename(columns={'new_feedid':'hist_feed_id'})
        train = train.merge(temp, on='new_user_id', how='left')
        test = test.merge(temp, on='new_user_id', how='left')

        temp = all_behavior.groupby(['new_user_id'])['new_authorid'].apply(list).reset_index()
        temp = temp.rename(columns={'new_authorid':'hist_author_id'})
        train = train.merge(temp, on='new_user_id', how='left')
        test = test.merge(temp, on='new_user_id', how='left')

    # 保存训练集、验证集、测试集
    train.reset_index(drop=True).to_csv(os.path.join(ROOT_PATH, "train_all_data_chusai.csv"), index=False)
    test.reset_index(drop=True).to_csv(os.path.join(ROOT_PATH, "test_data_chusai_b.csv"), index=False)


# 生成目录
submit_dir = os.path.join(ROOT_PATH, "single_model_result")
if not os.path.exists(submit_dir): os.makedirs(submit_dir)

model = os.path.join(ROOT_PATH, "model")
if not os.path.exists(model): os.makedirs(model)

pretrain = os.path.join(ROOT_PATH, "pretrain_model")
if not os.path.exists(pretrain): os.makedirs(pretrain)

submission = os.path.join(ROOT_PATH, "submission")
if not os.path.exists(submission): os.makedirs(submission)

# 数据集生成
prepare_data()