import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os
import pickle
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../model'))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from conf import *

test = pd.read_csv(os.path.join(ROOT_PATH, "test_data_chusai_b.csv"))
sparse_features = ['new_user_id', 'new_feedid', 'new_authorid']
dense_features = ['videoplayseconds']
USE_FEAT = sparse_features + dense_features + ['date_', 'description', 'manual_tag_list', 'manual_keyword_list']

train_all = pd.read_csv(os.path.join(ROOT_PATH, "train_all_data_chusai.csv"), usecols=USE_FEAT)

test = test[USE_FEAT]
train_all = train_all[USE_FEAT]
data_all = pd.concat((train_all, test)).reset_index(drop=True)

userid_list = data_all['new_user_id'].unique().tolist()
feed_list = data_all['new_feedid'].unique().tolist()
author_list = data_all['new_authorid'].unique().tolist()

mms = MinMaxScaler(feature_range=(0, 1))

USER_FEED_W2V_EMBEDDING_ARRAY = np.load(os.path.join(W2V_PATH, 'new_user_id_new_feedid_16_w2v.npy'))

FEED_USER_W2V_EMBEDDING_ARRAY =  np.load(os.path.join(W2V_PATH, 'new_feedid_new_user_id_16_w2v.npy'))

def Scalar(key, array, id_list, path, file_name):
    mms = MinMaxScaler(feature_range=(0, 1))
    prefix = 'feat_'
    temp = pd.DataFrame({prefix + str(i): array[:, i + 1]
                         for i in range(array.shape[-1] - 1)})
    temp[key] = array[:, 0]
    con_features = [prefix + str(i) for i in range(array.shape[-1] - 1)]
    # temp = reduce_mem_usages(temp)
#     temp = temp[temp[key].isin(id_list)].reset_index(drop=True)
    temp[con_features] = mms.fit_transform(temp[con_features])
    save_dict = {}
    for i in range(temp.shape[0]):
        save_dict[int(temp.loc[i, key])] = temp.loc[i, con_features].values.tolist()
    with open(os.path.join(USER_DATA_PATH, file_name), 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Scalar('new_feedid', TAG_TF_IDF, feed_list, PRETRAIN_PATH, 'manual_tag_tf_idf_8_scalar.pickle')
# Scalar('new_feedid', KEY_TF_IDF, feed_list, PRETRAIN_PATH, 'manual_key_tf_idf_8_scalar.pickle')
# Scalar('new_user_id', USER_TAG_TFIDF, userid_list, PRETRAIN_PATH, 'user_tag_tf_idf_8_scalar.pickle')

Scalar('new_feedid', USER_FEED_W2V_EMBEDDING_ARRAY, feed_list, W2V_PATH, 'userid_feedid_16_w2v_scalar.pickle')
# Scalar('new_feedid', AUTHOR_FEED_W2V_EMBEDDING_ARRAY, feed_list, W2V_PATH, 'authorid_feedid_16_w2v_scalar_new.pickle')
Scalar('new_user_id', FEED_USER_W2V_EMBEDDING_ARRAY, userid_list, W2V_PATH, 'feedid_userid_16_w2v_scalar.pickle')
# Scalar('new_user_id', AUTHOR_USER_W2V_EMBEDDING_ARRAY, userid_list, W2V_PATH, 'authorid_userid_16_w2v_scalar_new.pickle')
# Scalar('new_authorid', USER_AUTHOR_W2V_EMBEDDING_ARRAY, author_list, W2V_PATH, 'userid_authorid_16_w2v_scalar_new.pickle')

feed_info = pd.read_csv(FEED_INFO)
feed_info = feed_info.drop_duplicates(['feedid'], keep='first')
feed_info = feed_info.sort_values(by=['feedid'], ascending=True).reset_index(drop=True)
feed_info['new_feedid'] = list(range(feed_info.shape[0]))
feed_info = feed_info[
    ['new_feedid', 'videoplayseconds', 'description', 'manual_keyword_list', 'machine_keyword_list', 'manual_tag_list',
     'machine_tag_list']]
feed_info.description = feed_info.description.fillna('-1')


def get_tag(tag):
    tags = tag.split(";")
    tags = [t.split(" ") for t in tags]
    tags = [t[0] for t in tags if float(t[-1]) > 0.5]
    return ";".join(tags)


feed_info['manual_tag_list'] = feed_info['manual_tag_list'].fillna('-1')
feed_info['machine_tag_list'] = feed_info['machine_tag_list'].fillna('-1')
feed_info['manual_tag_list'] = feed_info.apply(
    lambda x: get_tag(x['machine_tag_list']) if x['manual_tag_list'] == '-1' else x['manual_tag_list'], axis=1)
feed_info['machine_tag_list'] = feed_info.apply(
    lambda x: x['manual_tag_list'] if x['machine_tag_list'] == '-1' else get_tag(x['machine_tag_list']), axis=1)

feed_info['manual_keyword_list'] = feed_info['manual_keyword_list'].fillna('-1')
feed_info['machine_keyword_list'] = feed_info['machine_keyword_list'].fillna('-1')
feed_info['manual_keyword_list'] = feed_info.apply(
    lambda x: x['machine_keyword_list'] if x['manual_keyword_list'] == '-1' else x['manual_keyword_list'], axis=1)
feed_info['machine_keyword_list'] = feed_info.apply(
    lambda x: x['manual_keyword_list'] if x['machine_keyword_list'] == '-1' else x['machine_keyword_list'], axis=1)


def split(x):
    key_ans = x.split(dem)
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))


key2index = {}
dem = ' '
description_list = list(map(split, feed_info['description'].values))
max_len = 256
# Notice : padding=`post`
description_list = pad_sequences(description_list, maxlen=max_len, padding='post')
save_dict = {}
print(len(key2index))
key = 'new_feedid'
file_name = 'feed_description.pickle'
for i in range(len(description_list)):
    save_dict[int(feed_info.loc[i, key])] = description_list[i]
with open(os.path.join(USER_DATA_PATH, file_name), 'wb') as handle:
    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

key2index = {}
dem = ';'
tag_list = list(map(split, feed_info['manual_tag_list'].values.tolist()))
max_len = 4
tag_list = pad_sequences(tag_list, maxlen=max_len, padding='post')
save_dict = {}
key = 'new_feedid'
file_name = 'feed_tag.pickle'
print("vacab size:", len(key2index))
for i in range(len(tag_list)):
    save_dict[int(feed_info.loc[i, key])] = tag_list[i]
with open(os.path.join(USER_DATA_PATH, file_name), 'wb') as handle:
    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

key2index = {}
dem = ';'
manual_keyword_list = list(map(split, feed_info['manual_keyword_list'].values.tolist()))
max_len = 4
manual_keyword_list = pad_sequences(manual_keyword_list, maxlen=max_len, padding='post')
save_dict = {}
key = 'new_feedid'
file_name = 'feed_keyword.pickle'
print("vacab size:", len(key2index))
for i in range(len(manual_keyword_list)):
    save_dict[int(feed_info.loc[i, key])] = manual_keyword_list[i]
with open(os.path.join(USER_DATA_PATH, file_name), 'wb') as handle:
    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

USE_DR = True
FEED_EMBEDDINGS_ARRAY = np.load(os.path.join(ROOT_PATH, 'new_feed_embedding.npy'))

if USE_DR:
    # 降维
    print("feed embedding reduce...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=16, random_state=42)
    FEED_EMBEDDINGS_ARRAY = np.concatenate((np.expand_dims(FEED_EMBEDDINGS_ARRAY[:,0], axis=1),
                                            pca.fit_transform(FEED_EMBEDDINGS_ARRAY[:,1:])), axis=1)

save_dict = {}
file_name = 'feed_embedding.pickle'
for i in range(FEED_EMBEDDINGS_ARRAY.shape[0]):
    save_dict[int(FEED_EMBEDDINGS_ARRAY[i, 0])] = FEED_EMBEDDINGS_ARRAY[i, 1:].flatten().tolist()
with open(os.path.join(USER_DATA_PATH, file_name), 'wb') as handle:
    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# feed_info = feed_info[feed_info['new_feedid'].isin(feed_list)].reset_index(drop=True)
feed_info['videoplayseconds'] = np.log(feed_info["videoplayseconds"] + 1.0)
feed_info['videoplayseconds'] = mms.fit_transform(np.expand_dims(feed_info['videoplayseconds'].values, axis=1))
save_dict = {}
key = 'new_feedid'
file_name = 'feed_videoplayseconds.pickle'
for i in range(feed_info.shape[0]):
    save_dict[int(feed_info.loc[i, key])] = feed_info.loc[i, 'videoplayseconds']
with open(os.path.join(USER_DATA_PATH, file_name), 'wb') as handle:
    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)