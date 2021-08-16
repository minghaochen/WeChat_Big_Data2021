# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os
from collections import OrderedDict
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../tools'))

sys.path.append(os.path.join(BASE_DIR, '..'))

import pickle
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../config'))
from conf import *
import time as ti

t1 = ti.time()
print("prepare input data...")
features_path = os.path.join(ROOT_PATH, f"user_data/features_dict")
if not os.path.exists(features_path): os.makedirs(features_path)

def generate_dict(file_name, day):
    con_features_data = pd.read_csv(os.path.join(features_path, file_name))
    con_features_data = con_features_data[con_features_data['date_']!=day]
    con_features = con_features_data.columns
    for f in con_features:
        if f != 'date_':
            feature_dict[f] = con_features_data[f].values

days = 15
# 生成特征字典
for day in tqdm(range(1, days+1)):
    feature_dict = OrderedDict()
    generate_dict("static.csv", day)

    prefix = 'tag_tf_idf_'
    generate_dict(f"{prefix}feature_data.csv", day)

    prefix = 'key_tf_idf_'
    generate_dict(f"{prefix}feature_data.csv", day)

    prefix = 'user_feed_15_day_'
    generate_dict(f"{prefix}feature_data.csv", day)

    prefix = 'multi_mode_feed_'
    generate_dict(f"{prefix}feature_data.csv", day)

    prefix = 'author_feed_15_day_'
    generate_dict(f"{prefix}feature_data.csv", day)

    prefix = 'feed_user_15_day_'
    generate_dict(f"{prefix}feature_data.csv", day)

    prefix = 'uer_tag_tfidf_'
    generate_dict(f"{prefix}feature_data.csv", day)

    prefix = 'author_user_15_day_'
    generate_dict(f"{prefix}feature_data.csv", day)

    prefix = 'user_author_15_day_'
    generate_dict(f"{prefix}feature_data.csv", day)

    with open(os.path.join(features_path, f"{day}.pickle"), 'wb') as handle:
        pickle.dump(feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

t2 = ti.time()
ts = (t2 - t1) / 3600.0
print(f"total time: {ts:.2f}")