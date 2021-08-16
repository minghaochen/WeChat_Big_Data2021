import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os
from scipy import sparse
from scipy.sparse.linalg import svds
import pickle

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

mms = MinMaxScaler(feature_range=(0, 1))

concat = data_all[['new_user_id', 'new_feedid']]
user_cnt = concat['new_user_id'].max() + 1
feed_cnt = concat['new_feedid'].max() + 1

### 1.构建交互稀疏矩阵
data = np.ones(len(concat))
user = concat['new_user_id'].values
feed_id = concat['new_feedid'].values
rating = sparse.coo_matrix((data, (user, feed_id)))
rating = (rating > 0) * 1.0

### 2.进行SVD分解
## svd for user-song pairs
n_component = 32
[u, s, vt] = svds(rating, k=n_component)
print(s[::-1])
s_feed = np.diag(s[::-1])

### 3.生成SVD特征向量
user_topics = pd.DataFrame(u[:, ::-1])
con_features = ['user_component_%d'%i for i in range(n_component)]
user_topics.columns = con_features
user_topics['new_user_id'] = range(user_cnt)
user_topics[con_features] = mms.fit_transform(user_topics[con_features])
save_dict = {}
file_name = 'user_topics.pickle'
key = 'new_user_id'
for i in range(user_topics.shape[0]):
    save_dict[int(user_topics.loc[i, key])] = user_topics.loc[i, con_features].values.tolist()
with open(os.path.join(USER_DATA_PATH, file_name), 'wb') as handle:
    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

feed_topics = pd.DataFrame(vt.transpose()[:, ::-1])
con_features = ['feed_component_%d'%i for i in range(n_component)]
feed_topics.columns = con_features
feed_topics['new_feedid'] = range(feed_cnt)
feed_topics[con_features] = mms.fit_transform(feed_topics[con_features])
save_dict = {}
file_name = 'feed_topics.pickle'
key = 'new_feedid'
for i in range(feed_topics.shape[0]):
    save_dict[int(feed_topics.loc[i, key])] = feed_topics.loc[i, con_features].values.tolist()
with open(os.path.join(USER_DATA_PATH, file_name), 'wb') as handle:
    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)