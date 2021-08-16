# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 22:35:12 2021

@author: mhchen
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import lightgbm
import numpy as np
from uauc import *
from sklearn.decomposition import PCA
import gensim
from utils import *
import gc
import time
import psutil
import datatable as dt
import os
info = psutil.virtual_memory()
print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
print(u'当前使用的总内存占比：',info.percent)

y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
# ## 采样训练
train1 = dt.fread('wbdc2021/data/wedata/wechat_algo_data1/user_action.csv').to_pandas()
train = dt.fread('wbdc2021/data/wedata/wechat_algo_data2/user_action.csv').to_pandas()
train =  pd.concat([train1, train]).reset_index(drop=True)
del train1 
gc.collect()
# 保留正样本
positive = train[(train['read_comment']==1) | (train['like']==1) |
                 (train['click_avatar']==1) | (train['forward']==1) | (train['favorite']==1) |
                 (train['comment']==1) | (train['follow']==1) ]
negative = train[~((train['read_comment']==1) | (train['like']==1) |
                   (train['click_avatar']==1) | (train['forward']==1) | (train['favorite']==1) |
                   (train['comment']==1) | (train['follow']==1))]
negative = negative.groupby('date_').apply(lambda sampler: sampler.sample(frac=0.30))
    
train = pd.concat([positive, negative]).reset_index(drop=True)
del positive,negative
gc.collect()

# 只选择7到14训练
# train = train[train['date_'] > 7].reset_index(drop=True)
# train = train.groupby('date_').apply(lambda sampler: sampler.sample(frac=0.80))

feed_info = pd.read_csv('wbdc2021/data/wedata/wechat_algo_data2/feed_info.csv')
feed_info = feed_info[['feedid', 'authorid','bgm_song_id', 'bgm_singer_id', 'videoplayseconds']]
train = train.merge(feed_info, on='feedid', how='left')

play_cols = ['is_finish', 'play_times', 'play', 'stay']
import pickle
def base1(df):
    for stat_cols in tqdm([['userid'],['feedid'],['authorid']]):
        stat_df = pd.read_csv(f'semi_feature/baseline_fea/{stat_cols}.csv')
        df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')
        del stat_df
        gc.collect()
    with open(f'semi_feature/baseline_fea/base_dict.pkl',"rb") as tf:
        base_dict = pickle.load(tf)
    for f in tqdm(['userid', 'feedid', 'authorid']):
        df[f + '_count'] = df[f].map(base_dict[f + '_count'])
    for f1, f2 in tqdm([['userid', 'feedid'],['userid', 'authorid']]):
        df['{}_in_{}_nunique'.format(f1, f2)] = df[f2].map(base_dict['{}_in_{}_nunique'.format(f1, f2)])
        df['{}_in_{}_nunique'.format(f2, f1)] = df[f1].map(base_dict['{}_in_{}_nunique'.format(f2, f1)])
    df['videoplayseconds_in_userid_mean'] = df['userid'].map(base_dict['videoplayseconds_in_userid_mean'])
    df['videoplayseconds_in_authorid_mean'] = df['authorid'].map(base_dict['videoplayseconds_in_authorid_mean'])
    df['feedid_in_authorid_nunique'] = df['authorid'].map(base_dict['feedid_in_authorid_nunique'])

    df['userid_authorid_count'] = df[['userid','authorid']].apply(tuple,axis=1)
    df['userid_authorid_count'] = df['userid_authorid_count'].map(base_dict['userid_authorid_count'])
    for f1, f2 in tqdm([['userid', 'authorid']]):
        df['{}_in_{}_count_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / (df[f2 + '_count'] + 1)
        df['{}_in_{}_count_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / (df[f1 + '_count'] + 1)
    
    del base_dict
    gc.collect()
    return df

train = base1(train)
print(train.shape)
reduce_mem_usage(train)

# 添加统计特征
print('----------------load statics ------------------------')
FEA_AGG_LIST = ['userid', 'feedid', 'authorid']

dict_feature = {}
statics_col = []
for column in FEA_AGG_LIST:
    column_feature = pd.read_csv(f'semi_feature/ctr_feature/10_len3_{column}_stastic.csv')
    column_feature = reduce_mem_usage(column_feature)
    dict_feature[column] = column_feature
    statics_col.extend(list(column_feature.columns))
    statics_col.remove(column)
statics_col = list(set(statics_col))
statics_col.remove('date_')

def merge_statics(df, mode='TRAIN'):
    if mode == 'TRAIN':
        df = df.query('date_>1').reset_index(drop=True)
    for column in FEA_AGG_LIST:
        print(column)
        start = time.time()
        column_feature = dict_feature[column]
        if mode == 'TRAIN':
            column_feature = column_feature.query('date_<15').reset_index(drop=True)
        else:
            df['date_']=15
            column_feature = column_feature.query('date_==15').reset_index(drop=True)
        df=df.merge(column_feature, on=[column, "date_"], how="left")
        end = time.time()
        print('-------------', column, f'time{end - start}--------------')
    return df

# 滑窗统计特征
train = merge_statics(train)
del dict_feature
gc.collect()

print('----------------load embedding ------------------------')
# 只挑选action为1的进行平均
EMB_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward", "favorite", "comment", "follow"]

dict_emb0 = {}
statics_col0 = []
for action in EMB_COLUMN_LIST:
    tmp_emb = pd.read_csv(f'semi_feature/embedding_agg/userid_{action}_mean.csv')
    tmp_emb = reduce_mem_usage(tmp_emb)
    dict_emb0[action] = tmp_emb
    statics_col0.extend(list(tmp_emb.columns))
statics_col0 = list(set(statics_col0))
statics_col0.remove('date_')
statics_col0.remove('userid')

def merge_emb_statics0(df, mode='TRAIN'):
    if mode == 'TRAIN':
        df = df.query('date_>1').reset_index(drop=True)
    for column in EMB_COLUMN_LIST:
        print(column)
        start = time.time()
        column_feature = dict_emb0[column]
        if mode == 'TRAIN':
            column_feature = column_feature.query('date_<15').reset_index(drop=True)
        else:
            df['date_'] = 15
            column_feature = column_feature.query('date_ == 15').reset_index(drop=True)
        df=df.merge(column_feature, on=['userid', "date_"], how="left")
        end = time.time()
        print('-------------', column, f'time{end - start}--------------')
    return df

train = merge_emb_statics0(train)
del dict_emb0
gc.collect()

print('-------------## 加载embedding--------------------------')
PCA_FLAG = False
VECTOR_SIZE = 16
def load_embedding(path='w2v/w2v.npy', pca_dim=0.99, column=None, prefix='', pca=PCA_FLAG):
    w2v_embedding = np.load(path)
    array_feedid = w2v_embedding[:, 0]
    w2v_embedding = w2v_embedding[:, 1:]
    original_columns_num = w2v_embedding.shape[1]
    if pca:
        pca = PCA(n_components=pca_dim)
        w2v_embedding = pca.fit_transform(w2v_embedding)
    columns_nums = w2v_embedding.shape[1]
    df_w2v = pd.DataFrame(w2v_embedding, columns=[f'{prefix}{column}_wv{i}' for i in range(columns_nums)])
    df_w2v[column] = array_feedid.astype(int)
    return reduce_mem_usage(df_w2v)

# tfidf
df_manual_keyfeedid_embedding = load_embedding(f'feature/w2v/manual_keyword_list_feedid_{VECTOR_SIZE}_w2v.npy',pca_dim=16, column='feedid',prefix='manual_key')
df_manual_tagfeedid_embedding = load_embedding(f'feature/w2v/manual_tag_list_feedid_{VECTOR_SIZE}_w2v.npy', pca_dim=16, column='feedid',prefix='manual_tag')
df_machine_keyfeedid_embedding = load_embedding(f'feature/w2v/machine_keyword_list_feedid_{VECTOR_SIZE}_w2v.npy',pca_dim=16, column='feedid',prefix='machine_key')
df_machine_tagfeedid_embedding = load_embedding(f'feature/w2v/machine_tag_list_feedid_{VECTOR_SIZE}_w2v.npy', pca_dim=16, column='feedid',prefix='machine_tag')

tfidf_col = list(df_manual_keyfeedid_embedding.columns)
tfidf_col += list(df_manual_tagfeedid_embedding.columns)
tfidf_col += list(df_machine_keyfeedid_embedding.columns)
tfidf_col += list(df_machine_tagfeedid_embedding.columns)
tfidf_col = list(set(tfidf_col))
tfidf_col.remove('feedid')

def merge_tfidf(df_tmp):
    df_tmp = df_tmp.merge(df_manual_keyfeedid_embedding, on='feedid', how='left')
    df_tmp = df_tmp.merge(df_manual_tagfeedid_embedding, on='feedid', how='left')
    df_tmp = df_tmp.merge(df_machine_keyfeedid_embedding, on='feedid', how='left')
    df_tmp = df_tmp.merge(df_machine_tagfeedid_embedding, on='feedid', how='left')
    return df_tmp

train = merge_tfidf(train)
del df_manual_keyfeedid_embedding,df_manual_tagfeedid_embedding,df_machine_keyfeedid_embedding,df_machine_tagfeedid_embedding
gc.collect()

# user维度的id
df_userid_embedding_by_feed = load_embedding(f'semi_feature/w2v/feedid_userid_{VECTOR_SIZE}_w2v.npy', 
                                             pca_dim=16, column='userid',prefix='feed')
df_userid_embedding_by_author = load_embedding(f'semi_feature/w2v/authorid_userid_{VECTOR_SIZE}_w2v.npy',
                                               pca_dim=16, column='userid',prefix='author')

userid_col = list(df_userid_embedding_by_feed.columns)
userid_col += list(df_userid_embedding_by_author.columns)
userid_col = list(set(userid_col))
userid_col.remove('userid')

def merge_uid(df_tmp):
    df_tmp = df_tmp.merge(df_userid_embedding_by_feed, on='userid', how='left')
    df_tmp = df_tmp.merge(df_userid_embedding_by_author, on='userid', how='left')
    return df_tmp

train = merge_uid(train)
del df_userid_embedding_by_feed,df_userid_embedding_by_author
gc.collect()

# feedid不同维度embedding
df_feedid_embedding_by_user = load_embedding(f'semi_feature/w2v/userid_feedid_{VECTOR_SIZE}_w2v.npy',
                                             pca_dim=16, column='feedid',prefix='user')
df_feedid_embedding_by_author = load_embedding(f'semi_feature/w2v/authorid_feedid_{VECTOR_SIZE}_w2v.npy',
                                               pca_dim=16, column='feedid',prefix='author')


feedid_col = list(df_feedid_embedding_by_user.columns)
feedid_col += list(df_feedid_embedding_by_author.columns)
feedid_col = list(set(feedid_col))
feedid_col.remove('feedid')

def merge_fid(df_tmp):
    df_tmp = df_tmp.merge(df_feedid_embedding_by_user, on='feedid', how='left')
    df_tmp = df_tmp.merge(df_feedid_embedding_by_author, on='feedid', how='left')
    return df_tmp

train = merge_fid(train)
del df_feedid_embedding_by_user,df_feedid_embedding_by_author
gc.collect()

info = psutil.virtual_memory()
print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
print(u'当前使用的总内存占比：',info.percent)

TRAIN_FEATURES = [f for f in train.columns if f not in ['date_'] + play_cols + y_list]

print('特征维度',len(TRAIN_FEATURES))
reduce_mem_usage(train)

train_x = train[train['date_'] < 14].reset_index(drop=True)
valid_x = train[train['date_'] == 14].reset_index(drop=True)
print('valid_day is 14')
# train_x = train[(train['date_'] < 11) | (train['date_'] > 11)].reset_index(drop=True)
# valid_x = train[train['date_'] == 11].reset_index(drop=True)

del train
gc.collect()

LIST_ACTION = y_list



from catboost import CatBoostClassifier
weight_auc = 0

dict_weight = {
            "read_comment": 4.0,  # 是否查看评论
            "like": 3.0,  # 是否点赞
            "click_avatar": 2.0,  # 是否点击头像
            "forward": 1.0,  # 是否转发
            "favorite": 1.0,
            "comment": 1.0,
            "follow": 1.0,
}

dict_uauc = {}
dict_feature_importance = {}
info = psutil.virtual_memory()
print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
print(u'当前使用的总内存占比：',info.percent)

model_dict = dict()
for action in LIST_ACTION:
    train_y =  train_x[action].values.astype(int)
    valid_y =  valid_x[action].values.astype(int)
#     model = CatBoostClassifier(iterations=50000, 
#                                task_type="GPU",
#                                eval_metric='AUC',
#                                loss_function='Logloss',
# #                                devices='0:1:2',
#                                devices=[0,1,2,3],
#                                learning_rate=0.07,
#                                class_weights = [1, 10],
#                                reg_lambda = 10,
#                               )
    
    model = CatBoostClassifier(iterations=70000, 
                               task_type="GPU",
                               eval_metric='AUC',
                               loss_function='Logloss',
                               devices=[0,1,2,3],
                               learning_rate=0.05,
#                                class_weights = [1, 5],
                               reg_lambda = 1,
                               od_type="Iter",
                               border_count=64,
                              )
    
    model.fit(train_x[TRAIN_FEATURES],train_y,
              eval_set=(valid_x[TRAIN_FEATURES], valid_y),
              verbose=500,
              early_stopping_rounds=500)
    y_preds = model.predict_proba(valid_x[TRAIN_FEATURES])[:, 1]
    uacu_valid, _ = uAUC(valid_y, y_preds, valid_x['userid'].values)
    weight_auc += uacu_valid * dict_weight[action]/10
    dict_uauc[action] = uacu_valid
#     model_dict[action] = model # 保存模型
    model.save_model(f"semi_model/valid14_4/model_{action}")
    del model 
    gc.collect()
#     y_test_preds = model.predict_proba(test[TRAIN_FEATURES])[:, 1]
#     sub[action] = y_test_preds
    print('*'*30, f'{action} best auc:{uacu_valid}', '*'*30)
#     dict_feature_importance[action] = sorted(zip(TRAIN_FEATURES, model.feature_importances_), key=lambda x:x[1], reverse=True)
print(f'weight_auc:{weight_auc}')

del train_x,valid_x
gc.collect()

### 测试集
test = pd.read_csv('wbdc2021/data/wedata/wechat_algo_data2/test_a.csv')
max_day = 15
test['date_'] = max_day
feed_info = pd.read_csv('wbdc2021/data/wedata/wechat_algo_data2/feed_info.csv')
feed_info = feed_info[['feedid', 'authorid','bgm_song_id', 'bgm_singer_id', 'videoplayseconds']]

test = test.merge(feed_info, on='feedid', how='left')
play_cols = ['is_finish', 'play_times', 'play', 'stay']
import pickle
def base1(df):
    for stat_cols in tqdm([['userid'],['feedid'],['authorid']]):
        stat_df = pd.read_csv(f'semi_feature/baseline_fea/{stat_cols}.csv')
        df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')
        del stat_df
        gc.collect()
    with open(f'semi_feature/baseline_fea/base_dict.pkl',"rb") as tf:
        base_dict = pickle.load(tf)
    for f in tqdm(['userid', 'feedid', 'authorid']):
        df[f + '_count'] = df[f].map(base_dict[f + '_count'])
    for f1, f2 in tqdm([['userid', 'feedid'],['userid', 'authorid']]):
        df['{}_in_{}_nunique'.format(f1, f2)] = df[f2].map(base_dict['{}_in_{}_nunique'.format(f1, f2)])
        df['{}_in_{}_nunique'.format(f2, f1)] = df[f1].map(base_dict['{}_in_{}_nunique'.format(f2, f1)])
    df['videoplayseconds_in_userid_mean'] = df['userid'].map(base_dict['videoplayseconds_in_userid_mean'])
    df['videoplayseconds_in_authorid_mean'] = df['authorid'].map(base_dict['videoplayseconds_in_authorid_mean'])
    df['feedid_in_authorid_nunique'] = df['authorid'].map(base_dict['feedid_in_authorid_nunique'])

    df['userid_authorid_count'] = df[['userid','authorid']].apply(tuple,axis=1)
    df['userid_authorid_count'] = df['userid_authorid_count'].map(base_dict['userid_authorid_count'])
    for f1, f2 in tqdm([['userid', 'authorid']]):
        df['{}_in_{}_count_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / (df[f2 + '_count'] + 1)
        df['{}_in_{}_count_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / (df[f1 + '_count'] + 1)
    
    del base_dict
    gc.collect()
    return df

test = base1(test)
reduce_mem_usage(test)

print('----------------load statics ------------------------')
FEA_AGG_LIST = ['userid', 'feedid', 'authorid']

dict_feature = {}
statics_col = []
for column in FEA_AGG_LIST:
    column_feature = pd.read_csv(f'semi_feature/ctr_feature/10_len3_{column}_stastic.csv')
    column_feature = reduce_mem_usage(column_feature)
    dict_feature[column] = column_feature
    statics_col.extend(list(column_feature.columns))
    statics_col.remove(column)
statics_col = list(set(statics_col))
statics_col.remove('date_')

def merge_statics(df, mode='TRAIN'):
    if mode == 'TRAIN':
        df = df.query('date_>1').reset_index(drop=True)
    for column in FEA_AGG_LIST:
        print(column)
        start = time.time()
        column_feature = dict_feature[column]
        if mode == 'TRAIN':
            column_feature = column_feature.query('date_<15').reset_index(drop=True)
        else:
            df['date_']=15
            column_feature = column_feature.query('date_==15').reset_index(drop=True)
        df=df.merge(column_feature, on=[column, "date_"], how="left")
        end = time.time()
        print('-------------', column, f'time{end - start}--------------')
    return df

# 滑窗统计特征
test = merge_statics(test, 'TEST')
del dict_feature
gc.collect()

print('----------------load embedding ------------------------')
# 只挑选action为1的进行平均
EMB_COLUMN_LIST = ["read_comment", "like", "click_avatar",  "forward", "favorite", "comment", "follow"]

dict_emb0 = {}
statics_col0 = []
for action in EMB_COLUMN_LIST:
    tmp_emb = pd.read_csv(f'semi_feature/embedding_agg/userid_{action}_mean.csv')
    tmp_emb = reduce_mem_usage(tmp_emb)
    dict_emb0[action] = tmp_emb
    statics_col0.extend(list(tmp_emb.columns))
statics_col0 = list(set(statics_col0))
statics_col0.remove('date_')
statics_col0.remove('userid')

def merge_emb_statics0(df, mode='TRAIN'):
    if mode == 'TRAIN':
        df = df.query('date_>1').reset_index(drop=True)
    for column in EMB_COLUMN_LIST:
        print(column)
        start = time.time()
        column_feature = dict_emb0[column]
        if mode == 'TRAIN':
            column_feature = column_feature.query('date_<15').reset_index(drop=True)
        else:
            df['date_'] = 15
            column_feature = column_feature.query('date_ == 15').reset_index(drop=True)
        df=df.merge(column_feature, on=['userid', "date_"], how="left")
        end = time.time()
        print('-------------', column, f'time{end - start}--------------')
    return df

test = merge_emb_statics0(test, 'TEST')
del dict_emb0
gc.collect()

print('-------------## 加载embedding--------------------------')
PCA_FLAG = False
VECTOR_SIZE = 16
def load_embedding(path='w2v/w2v.npy', pca_dim=0.99, column=None, prefix='', pca=PCA_FLAG):
    w2v_embedding = np.load(path)
    array_feedid = w2v_embedding[:, 0]
    w2v_embedding = w2v_embedding[:, 1:]
    original_columns_num = w2v_embedding.shape[1]
    if pca:
        pca = PCA(n_components=pca_dim)
        w2v_embedding = pca.fit_transform(w2v_embedding)
    columns_nums = w2v_embedding.shape[1]
    df_w2v = pd.DataFrame(w2v_embedding, columns=[f'{prefix}{column}_wv{i}' for i in range(columns_nums)])
    df_w2v[column] = array_feedid.astype(int)
    return reduce_mem_usage(df_w2v)

# tfidf
df_manual_keyfeedid_embedding = load_embedding(f'feature/w2v/manual_keyword_list_feedid_{VECTOR_SIZE}_w2v.npy',pca_dim=16, column='feedid',prefix='manual_key')
df_manual_tagfeedid_embedding = load_embedding(f'feature/w2v/manual_tag_list_feedid_{VECTOR_SIZE}_w2v.npy', pca_dim=16, column='feedid',prefix='manual_tag')
df_machine_keyfeedid_embedding = load_embedding(f'feature/w2v/machine_keyword_list_feedid_{VECTOR_SIZE}_w2v.npy',pca_dim=16, column='feedid',prefix='machine_key')
df_machine_tagfeedid_embedding = load_embedding(f'feature/w2v/machine_tag_list_feedid_{VECTOR_SIZE}_w2v.npy', pca_dim=16, column='feedid',prefix='machine_tag')

tfidf_col = list(df_manual_keyfeedid_embedding.columns)
tfidf_col += list(df_manual_tagfeedid_embedding.columns)
tfidf_col += list(df_machine_keyfeedid_embedding.columns)
tfidf_col += list(df_machine_tagfeedid_embedding.columns)
tfidf_col = list(set(tfidf_col))
tfidf_col.remove('feedid')

def merge_tfidf(df_tmp):
    df_tmp = df_tmp.merge(df_manual_keyfeedid_embedding, on='feedid', how='left')
    df_tmp = df_tmp.merge(df_manual_tagfeedid_embedding, on='feedid', how='left')
    df_tmp = df_tmp.merge(df_machine_keyfeedid_embedding, on='feedid', how='left')
    df_tmp = df_tmp.merge(df_machine_tagfeedid_embedding, on='feedid', how='left')
    return df_tmp


test = merge_tfidf(test)
del df_manual_keyfeedid_embedding,df_manual_tagfeedid_embedding,df_machine_keyfeedid_embedding,df_machine_tagfeedid_embedding
gc.collect()

# user维度的id
df_userid_embedding_by_feed = load_embedding(f'semi_feature/w2v/feedid_userid_{VECTOR_SIZE}_w2v.npy', 
                                             pca_dim=16, column='userid',prefix='feed')
df_userid_embedding_by_author = load_embedding(f'semi_feature/w2v/authorid_userid_{VECTOR_SIZE}_w2v.npy',
                                               pca_dim=16, column='userid',prefix='author')

userid_col = list(df_userid_embedding_by_feed.columns)
userid_col += list(df_userid_embedding_by_author.columns)
userid_col = list(set(userid_col))
userid_col.remove('userid')

def merge_uid(df_tmp):
    df_tmp = df_tmp.merge(df_userid_embedding_by_feed, on='userid', how='left')
    df_tmp = df_tmp.merge(df_userid_embedding_by_author, on='userid', how='left')
    return df_tmp

test = merge_uid(test)
del df_userid_embedding_by_feed,df_userid_embedding_by_author
gc.collect()

# feedid不同维度embedding
df_feedid_embedding_by_user = load_embedding(f'semi_feature/w2v/userid_feedid_{VECTOR_SIZE}_w2v.npy',
                                             pca_dim=16, column='feedid',prefix='user')
df_feedid_embedding_by_author = load_embedding(f'semi_feature/w2v/authorid_feedid_{VECTOR_SIZE}_w2v.npy',
                                               pca_dim=16, column='feedid',prefix='author')


feedid_col = list(df_feedid_embedding_by_user.columns)
feedid_col += list(df_feedid_embedding_by_author.columns)
feedid_col = list(set(feedid_col))
feedid_col.remove('feedid')

def merge_fid(df_tmp):
    df_tmp = df_tmp.merge(df_feedid_embedding_by_user, on='feedid', how='left')
    df_tmp = df_tmp.merge(df_feedid_embedding_by_author, on='feedid', how='left')
    return df_tmp

test = merge_fid(test)
del df_feedid_embedding_by_user,df_feedid_embedding_by_author
gc.collect()

reduce_mem_usage(test)

info = psutil.virtual_memory()
print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
print(u'当前使用的总内存占比：',info.percent)

sub = pd.DataFrame()
sub["userid"] = test["userid"]
sub["feedid"] = test["feedid"]
for action in LIST_ACTION:
#     model = model_dict[action]
    model = CatBoostClassifier()
    model.load_model(f"semi_model/valid14_4/model_{action}")
    sub[action] = model.predict_proba(test[TRAIN_FEATURES])[:, 1]
    
sub.to_csv(f'upload/semi_sub_weight_auc:{weight_auc*100:.2f}.csv', index=False)
print('预测结束')