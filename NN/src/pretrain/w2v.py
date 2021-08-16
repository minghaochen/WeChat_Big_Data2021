import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from conf import *

VECTOR_SIZE = 16


def get_w2v(all_behavior, column1='userid', column2='authorid'):
    assert column1 in all_behavior.columns
    assert column2 in all_behavior.columns
    texts = all_behavior.groupby(column1)[column2].apply(list)
    model = Word2Vec(sentences=texts, vector_size=VECTOR_SIZE, window=5, sg=1, min_count=1, workers=10, epochs=15, seed=2021)
    w2v = np.zeros([len(model.wv.index_to_key), VECTOR_SIZE])
    for i, index_id in enumerate(model.wv.index_to_key):
        w2v[i] = model.wv[int(index_id)]
    array_id = np.array(model.wv.index_to_key)
    w2v = np.concatenate([array_id[:, None], w2v], axis=-1)
    np.save(os.path.join(W2V_PATH, f"{column1}_{column2}_{VECTOR_SIZE}_w2v"), w2v)

if __name__ == '__main__':
    print("*"*20+"训练词向量"+"*"*20)
    target = ["read_comment", "like", "click_avatar", "forward", 'favorite', 'comment', 'follow']
    test = pd.read_csv(os.path.join(ROOT_PATH, "test_data_chusai_a.csv"))
    test_b = pd.read_csv(os.path.join(ROOT_PATH, "test_data_chusai_b.csv"))
    sparse_features = ['new_user_id', 'new_feedid', 'new_authorid']
    dense_features = ['videoplayseconds']
    USE_FEAT = sparse_features + dense_features  + ['date_', 'description', 'manual_tag_list', 'manual_keyword_list']

    train_all = pd.read_csv(os.path.join(ROOT_PATH, "train_all_data_chusai.csv"), usecols=USE_FEAT)
    test = test[USE_FEAT]
    test_b = test_b[USE_FEAT]
    test = pd.concat((test, test_b), axis=0)
    train_all = train_all[USE_FEAT]
    all_behavior = pd.concat((train_all, test)).reset_index(drop=True)

    list_embedding_column = ['new_user_id', 'new_feedid', 'new_authorid']
    for column1 in list_embedding_column:
        for column2 in ['new_user_id', 'new_feedid', 'new_authorid']:
            if column1 != column2:
                print(column1 +'_'+column2)
                get_w2v(all_behavior, column1, column2)
