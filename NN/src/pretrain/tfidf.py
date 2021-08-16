import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import os
import sys
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from conf import *


if __name__ == '__main__':
    print("*" * 20 + "训练TFIDF" + "*" * 20)
    target = ["read_comment", "like", "click_avatar", "forward", 'favorite', 'comment', 'follow']
    test_b = pd.read_csv(os.path.join(ROOT_PATH, "test_data_chusai_b.csv"))
    sparse_features = ['new_user_id', 'new_feedid', 'new_authorid']
    dense_features = ['videoplayseconds', 'device']
    USE_FEAT = sparse_features + dense_features + ['date_', 'description', 'manual_tag_list', 'manual_keyword_list']

    train_all = pd.read_csv(os.path.join(ROOT_PATH, "train_all_data_chusai.csv"), usecols=USE_FEAT)
    test_b = test_b[USE_FEAT]
    train_all = train_all[USE_FEAT]
    all_behavior = pd.concat((train_all, test_b)).reset_index(drop=True)
    all_behavior['description'] = all_behavior['description'].fillna("-1")
    all_behavior['manual_tag_list'] = all_behavior['manual_tag_list'].fillna("-1")
    all_behavior['manual_keyword_list'] = all_behavior['manual_keyword_list'].fillna("-1")


    temp = all_behavior.groupby(['new_user_id'])['new_feedid'].apply(list).reset_index()
    text = temp['new_feedid'].values.tolist()
    text = [' '.join([str(j) for j in i]) for i in text]
    tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
    tv_fit = tv.fit_transform(text)
    x = tv_fit.toarray()
    pca = PCA(n_components=64, random_state=42)
    array = np.concatenate((np.expand_dims(temp['new_user_id'].values, axis=1),
                            pca.fit_transform(x)), axis=1)
    key = 'new_user_id'
    prefix = 'user_feed_tfidf_'
    temp = pd.DataFrame({prefix + str(i): array[:, i + 1]
                         for i in range(array.shape[-1] - 1)})
    temp[key] = array[:, 0]
    con_features = [prefix + str(i) for i in range(array.shape[-1] - 1)]
    save_dict = {}
    file_name = 'user_feed_tfidf.pickle'
    for i in range(temp.shape[0]):
        save_dict[int(temp.loc[i, key])] = temp.loc[i, con_features].values.tolist()
    with open('../../data/user_data/'+file_name, 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    temp = all_behavior.groupby(['new_feedid'])['new_user_id'].apply(list).reset_index()
    text = temp['new_user_id'].values.tolist()
    text = [' '.join([str(j) for j in i]) for i in text]
    tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
    tv_fit = tv.fit_transform(text)
    x = tv_fit.toarray()
    pca = PCA(n_components=64, random_state=42)
    array = np.concatenate((np.expand_dims(temp['new_feedid'].values, axis=1),
                            pca.fit_transform(x)), axis=1)
    key = 'new_feedid'
    prefix = 'feed_user_tfidf_'
    temp = pd.DataFrame({prefix + str(i): array[:, i + 1]
                         for i in range(array.shape[-1] - 1)})
    temp[key] = array[:, 0]
    con_features = [prefix + str(i) for i in range(array.shape[-1] - 1)]
    save_dict = {}
    file_name = 'feed_user_tfidf.pickle'
    for i in range(temp.shape[0]):
        save_dict[int(temp.loc[i, key])] = temp.loc[i, con_features].values.tolist()
    with open('../../data/user_data/'+file_name, 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # feed tag/key tf-idf
    # feed_info = pd.read_csv(FEED_INFO)
    # feed_info = feed_info.drop_duplicates(['feedid'], keep='first')
    # feed_info = feed_info.sort_values(by=['feedid'], ascending=True)
    # feed_info['new_feedid'] = list(range(feed_info.shape[0]))
    # feed_info = feed_info[['new_feedid', 'manual_keyword_list', 'manual_tag_list']]
    # feed_info.manual_tag_list = feed_info.manual_tag_list.fillna('-1')
    # feed_info.manual_keyword_list = feed_info.manual_keyword_list.fillna('-1')
    #
    # texts = feed_info['manual_keyword_list'].apply(lambda x: " ".join(x.split(";")))
    # tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
    # tv_fit = tv.fit_transform(texts)
    # x = tv_fit.toarray()
    # pca = PCA(n_components=16, random_state=42)
    # np.save(os.path.join(PRETRAIN_PATH, 'new_manual_key_tf_idf_16.npy'),
    #         np.concatenate((np.expand_dims(feed_info['new_feedid'].values, axis=1),
    #                         pca.fit_transform(x)), axis=1))
    #
    # texts = feed_info['manual_tag_list'].apply(lambda x: " ".join(x.split(";")))
    # tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
    # tv_fit = tv.fit_transform(texts)
    # x = tv_fit.toarray()
    # pca = PCA(n_components=16, random_state=42)
    # np.save(os.path.join(PRETRAIN_PATH, 'new_manual_tag_tf_idf_16.npy'),
    #         np.concatenate((np.expand_dims(feed_info['new_feedid'].values, axis=1),
    #                         pca.fit_transform(x)), axis=1))
