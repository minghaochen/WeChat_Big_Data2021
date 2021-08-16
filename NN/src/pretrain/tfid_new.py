from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import os
import sys
import gc
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from conf import *

target = ["read_comment", "like", "click_avatar", "forward", 'favorite', 'comment', 'follow']
# test = pd.read_csv(os.path.join(ROOT_PATH, "test_data_chusai_a.csv"))
test_b = pd.read_csv(os.path.join(ROOT_PATH, "test_data_chusai_b.csv"))
sparse_features = ['new_user_id', 'new_feedid', 'new_authorid']
dense_features = ['videoplayseconds', 'device']
USE_FEAT = sparse_features + dense_features + ['date_', 'description', 'manual_tag_list', 'manual_keyword_list']

train_all = pd.read_csv(os.path.join(ROOT_PATH, "train_all_data_chusai.csv"), usecols=USE_FEAT)
test_b = test_b[USE_FEAT]
train_all = train_all[USE_FEAT]
df_train_val = pd.concat((train_all, test_b)).reset_index(drop=True)

def tfidf_svd(data, f1, f2, n_components=64):
    tmp     = data.groupby(f1, as_index=False)[f2].agg({'list': lambda x: ' '.join(list(x.astype('str')))})
    tfidf   = TfidfVectorizer(max_df=0.95, min_df=3, sublinear_tf=True)
    res     = tfidf.fit_transform(tmp['list'])
    print('svd start')
    svd     = TruncatedSVD(n_components=n_components, random_state=2021)
    svd_res = svd.fit_transform(res)
    print('svd finished')
    for i in (range(n_components)):
        tmp['{}_{}_tfidf_svd_{}'.format(f1, f2, i)] = svd_res[:, i]
        tmp['{}_{}_tfidf_svd_{}'.format(f1, f2, i)] = tmp['{}_{}_tfidf_svd_{}'.format(f1, f2, i)].astype(np.float32)
    del tmp['list']
    return tmp

def get_first_svd_features(f1_, n_components = 64):
    # userid_id_dic : 用户的mapping字典；
    # first_cls_dic[f1_] : f1的mapping字典；
    f1_embedding_userid = tfidf_svd(df_train_val[[f1_, 'new_user_id']], f1_, 'new_user_id', n_components)
    f1_embedding_userid = f1_embedding_userid.fillna(0)
    save_dict = {}
    file_name = f'{f1_}_userid_tfidf.pickle'
    key = f1_
    con_features = [c for c in f1_embedding_userid.columns if c not in [f1_]]
    for i in range(f1_embedding_userid.shape[0]):
        save_dict[int(f1_embedding_userid.loc[i, key])] = f1_embedding_userid.loc[i, con_features].values.tolist()
    with open(os.path.join(USER_DATA_PATH, file_name), 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del save_dict, f1_embedding_userid
    gc.collect()

    userid_embedding_f1 = tfidf_svd(df_train_val[['new_user_id', f1_]], 'new_user_id', f1_, n_components)
    userid_embedding_f1 = userid_embedding_f1.fillna(0)
    save_dict = {}
    file_name = f'userid_{f1_}_tfidf.pickle'
    key = 'new_user_id'
    con_features = [c for c in userid_embedding_f1.columns if c not in ['new_user_id']]
    for i in range(userid_embedding_f1.shape[0]):
        save_dict[int(userid_embedding_f1.loc[i, key])] = userid_embedding_f1.loc[i, con_features].values.tolist()
    with open(os.path.join(USER_DATA_PATH, file_name), 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del save_dict, userid_embedding_f1
    gc.collect()

get_first_svd_features('new_feedid')
get_first_svd_features('new_authorid')