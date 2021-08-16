import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os
import pickle
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../tools'))
from utils import reduce_mem_usages

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from conf import *

print("prepare static feature...")

target = ["read_comment", "like", "click_avatar", "forward"]
test = pd.read_csv(os.path.join(ROOT_PATH, "test_data_chusai_b.csv"))
sparse_features = ['new_user_id', 'new_feedid', 'new_authorid', 'new_bgm_song_id']
USE_FEAT = sparse_features + ['date_']

train_all = pd.read_csv(os.path.join(ROOT_PATH, "train_all_data_chusai.csv"), usecols=USE_FEAT)
test = test[USE_FEAT]
train_all = train_all[USE_FEAT]
data_all = pd.concat((train_all, test)).reset_index(drop=True)
data_all['cnt'] = 1

userid_list = data_all['new_user_id'].unique().tolist()
feed_list = data_all['new_feedid'].unique().tolist()
author_list = data_all['new_authorid'].unique().tolist()

mms = MinMaxScaler(feature_range=(0, 1))

keys = ['new_authorid', 'date_']
temp = data_all.groupby(keys).agg(count=('cnt', 'count')).reset_index()
temp = reduce_mem_usages(temp)
temp['count'] = mms.fit_transform(np.expand_dims(temp['count'].values, 1))
save_dict = {}
for i in range(temp.shape[0]):
    if int(temp.loc[i, 'new_authorid']) not in save_dict.keys():
        save_dict[int(temp.loc[i, 'new_authorid'])] = {}
    save_dict[int(temp.loc[i, 'new_authorid'])][int(temp.loc[i, 'date_'])] = temp.loc[i, 'count']
file_name = "_".join(keys)+"_count.pickle"
with open('../../data/user_data/' + file_name, 'wb') as handle:
    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# keys = ['new_feedid', 'date_']
# temp = data_all.groupby(keys).agg(count=('cnt', 'count')).reset_index()
# temp = reduce_mem_usages(temp)
# temp['count'] = mms.fit_transform(np.expand_dims(temp['count'].values, 1))
# save_dict = {}
# for i in range(temp.shape[0]):
#     if int(temp.loc[i, 'new_feedid']) not in save_dict.keys():
#         save_dict[int(temp.loc[i, 'new_feedid'])] = {}
#     save_dict[int(temp.loc[i, 'new_feedid'])][int(temp.loc[i, 'date_'])] = temp.loc[i, 'count']
# file_name = "_".join(keys)+"_count.pickle"
# with open('../../data/user_data/' + file_name, 'wb') as handle:
#     pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# keys = ['new_user_id', 'date_']
# temp = data_all.groupby(keys).agg(count=('cnt', 'count')).reset_index()
# temp = reduce_mem_usages(temp)
# temp['count'] = mms.fit_transform(np.expand_dims(temp['count'].values, 1))
# save_dict = {}
# for i in range(temp.shape[0]):
#     if int(temp.loc[i, 'new_user_id']) not in save_dict.keys():
#         save_dict[int(temp.loc[i, 'new_user_id'])] = {}
#     save_dict[int(temp.loc[i, 'new_user_id'])][int(temp.loc[i, 'date_'])] = temp.loc[i, 'count']
# file_name = "_".join(keys)+"_count.pickle"
# with open('../../data/user_data/' + file_name, 'wb') as handle:
#     pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# keys = ['new_user_id', 'new_authorid', 'date_']
# temp = data_all.groupby(keys).agg(count=('cnt', 'count')).reset_index()
# temp = reduce_mem_usages(temp)
# temp['count'] = mms.fit_transform(np.expand_dims(temp['count'].values, 1))
# save_dict = {}
# for i in range(temp.shape[0]):
#     if int(temp.loc[i, 'new_user_id']) not in save_dict.keys():
#         save_dict[int(temp.loc[i, 'new_user_id'])] = {}
#     if int(temp.loc[i, 'new_authorid']) not in save_dict[int(temp.loc[i, 'new_user_id'])].keys():
#         save_dict[int(temp.loc[i, 'new_user_id'])][int(temp.loc[i, 'new_authorid'])] = {}
#     save_dict[int(temp.loc[i, 'new_user_id'])][int(temp.loc[i, 'new_authorid'])][int(temp.loc[i, 'date_'])] = temp.loc[i, 'count']
# file_name = "_".join(keys)+"_count.pickle"
# with open('../../data/user_data/' + file_name, 'wb') as handle:
#     pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# keys = ['new_user_id', 'new_feedid']
# temp = data_all.groupby(keys).agg(count=('cnt', 'count')).reset_index()
# temp = reduce_mem_usages(temp)
# temp['count'] = mms.fit_transform(np.expand_dims(temp['count'].values, 1))
# save_dict = {}
# for i in range(temp.shape[0]):
#     if int(temp.loc[i, 'new_user_id']) not in save_dict.keys():
#         save_dict[int(temp.loc[i, 'new_user_id'])] = {}
#     save_dict[int(temp.loc[i, 'new_user_id'])][int(temp.loc[i, 'new_feedid'])] = temp.loc[i, 'count']
# file_name = "_".join(keys)+"_count.pickle"
# with open('../../data/user_data/' + file_name, 'wb') as handle:
#     pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# keys = ['new_user_id', 'new_bgm_song_id']
# temp = data_all.groupby(keys).agg(count=('cnt', 'count')).reset_index()
# temp = reduce_mem_usages(temp)
# temp['count'] = mms.fit_transform(np.expand_dims(temp['count'].values, 1))
# save_dict = {}
# for i in range(temp.shape[0]):
#     if int(temp.loc[i, 'new_user_id']) not in save_dict.keys():
#         save_dict[int(temp.loc[i, 'new_user_id'])] = {}
#     save_dict[int(temp.loc[i, 'new_user_id'])][int(temp.loc[i, 'new_bgm_song_id'])] = temp.loc[i, 'count']
# file_name = "_".join(keys)+"_count.pickle"
# with open('../../data/user_data/' + file_name, 'wb') as handle:
#     pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# keys = ['new_user_id', 'new_authorid']
# temp = data_all.groupby(keys).agg(count=('cnt', 'count')).reset_index()
# temp = reduce_mem_usages(temp)
# temp['count'] = mms.fit_transform(np.expand_dims(temp['count'].values, 1))
# save_dict = {}
# for i in range(temp.shape[0]):
#     if int(temp.loc[i, 'new_user_id']) not in save_dict.keys():
#         save_dict[int(temp.loc[i, 'new_user_id'])] = {}
#     save_dict[int(temp.loc[i, 'new_user_id'])][int(temp.loc[i, 'new_authorid'])] = temp.loc[i, 'count']
# file_name = "_".join(keys)+"_count.pickle"
# with open('../../data/user_data/' + file_name, 'wb') as handle:
#     pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print("finished")