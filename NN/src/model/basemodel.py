import sys
import os
import pickle

import torch.nn.init

from attack import *
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../third_party'))
from deepctr_torch.models.basemodel import *
from deepctr_torch.layers.sequence import SequencePoolingLayer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    # for varlen sparse features, {embedding_name: nn.EmbeddingBag}
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse)
         for feat in
         sparse_feature_columns + varlen_sparse_feature_columns}
    )

    # for feat in varlen_sparse_feature_columns:
    #     embedding_dict[feat.embedding_name] = nn.EmbeddingBag(
    #         feat.dimension, embedding_size, sparse=sparse, mode=feat.combiner)

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict.to(device)

with open('../../data/user_data/feed_videoplayseconds.pickle', 'rb') as handle:
    videoplayseconds_dict = pickle.load(handle)

with open('../../data/user_data/new_feedid_date__count.pickle', 'rb') as handle:
    feed_date_count_dict = pickle.load(handle)

with open('../../data/user_data/new_user_id_date__count.pickle', 'rb') as handle:
    user_date_count_dict = pickle.load(handle)

with open('../../data/user_data/feed_embedding.pickle', 'rb') as handle:
    feed_embedding_dict = pickle.load(handle)

with open('../../data/user_data/userid_feedid_16_w2v_scalar.pickle', 'rb') as handle:
    user_feed_embedding_dict = pickle.load(handle)

with open('../../data/user_data/feedid_userid_16_w2v_scalar.pickle', 'rb') as handle:
    feed_user_embedding_dict = pickle.load(handle)


with open('../../data/user_data/userid_new_authorid_tfidf.pickle', 'rb') as handle:
    user_author_tfidf_dict = pickle.load(handle)

with open('../../data/user_data/new_authorid_userid_tfidf.pickle', 'rb') as handle:
    author_user_tfidf_dict = pickle.load(handle)

with open('../../data/user_data/userid_new_feedid_tfidf.pickle', 'rb') as handle:
    user_feed_tfidf_dict = pickle.load(handle)

with open('../../data/user_data/new_feedid_userid_tfidf.pickle', 'rb') as handle:
    feed_user_tfidf_dict = pickle.load(handle)

with open('../../data/user_data/user_topics.pickle', 'rb') as handle:
    user_topics_dict = pickle.load(handle)

with open('../../data/user_data/feed_topics.pickle', 'rb') as handle:
    feed_topics_dict = pickle.load(handle)


with open('../../data/user_data/feed_description.pickle', 'rb') as handle:
    description_dict = pickle.load(handle)

with open('../../data/user_data/feed_tag.pickle', 'rb') as handle:
    tag_dict = pickle.load(handle)

with open('../../data/user_data/feed_keyword.pickle', 'rb') as handle:
    keyword_dict = pickle.load(handle)

class MyDataset(Data.TensorDataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, x, y, date, feature_index):
        print("init dateset...")
        self.x = x
        self.y = y
        self.feature_index = feature_index
        self.date = date

        self.userid = self.x[:, self.feature_index['new_user_id'][0]:self.feature_index['new_user_id'][1]].squeeze()
        self.feedid = self.x[:, self.feature_index['new_feedid'][0]:self.feature_index['new_feedid'][1]].squeeze()
        self.authorid = self.x[:, self.feature_index['new_authorid'][0]:self.feature_index['new_authorid'][1]].squeeze()
        self.bgmid = self.x[:, self.feature_index['new_bgm_song_id'][0]:self.feature_index['new_bgm_song_id'][1]].squeeze()

        self.videoplayseconds = videoplayseconds_dict


    def __getitem__(self, index):

        feedid = int(self.feedid[index])
        userid = int(self.userid[index])
        authorid = int(self.authorid[index])
        date = int(self.date[index])

        feature_list = []

        videoplayseconds = videoplayseconds_dict[feedid]
        feature_list.append(torch.tensor([videoplayseconds]))

        feed_date_count = feed_date_count_dict[feedid][date]
        feature_list.append(torch.tensor([feed_date_count]))

        user_date_count = user_date_count_dict[userid][date]
        feature_list.append(torch.tensor([user_date_count]))

        user_feed_embedding = user_feed_embedding_dict[feedid]
        feature_list.append(torch.tensor(user_feed_embedding))

        multi_mode_feed_embedding = feed_embedding_dict[feedid]
        feature_list.append(torch.tensor(multi_mode_feed_embedding))

        feed_user_embedding = feed_user_embedding_dict[userid]
        feature_list.append(torch.tensor(feed_user_embedding))

        try:
            user_feed_tfidf = user_feed_tfidf_dict[userid]
        except:
            user_feed_tfidf = [0.]*64
        finally:
            feature_list.append(torch.tensor(user_feed_tfidf))

        try:
            feed_user_tfidf = feed_user_tfidf_dict[feedid]
        except:
            feed_user_tfidf = [0.]*64
        finally:
            feature_list.append(torch.tensor(feed_user_tfidf))

        description = description_dict[feedid]
        feature_list.append(torch.tensor(description))

        tag = tag_dict[feedid]
        feature_list.append(torch.tensor(tag))

        keyword = keyword_dict[feedid]
        feature_list.append(torch.tensor(keyword))

        if self.y != None:
            return (torch.cat([self.x[index]]+feature_list), self.y[index])
        else:
            return (torch.cat([self.x[index]]+feature_list), torch.tensor([0]))

    def __len__(self):
        return self.x.size(0)


class MyBaseModel(BaseModel):

    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):

        super(BaseModel, self).__init__()
        torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns
        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(
                "`gpus[0]` should be the same gpu with `device`")

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)

        # 加载预训练embedding
        # self.init_embedding("new_feedid", USER_FEED_W2V_EMBEDDING)
        # self.init_embedding("new_user_id", FEED_USER_W2V_EMBEDDING_)
        # self.init_embedding("new_authorid", USER_AUTHOR_W2V_EMBEDDING)

        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=True) for feat in
        #              self.dnn_feature_columns}
        #         )
        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device)
        self.linear_model_1 = Linear(
            linear_feature_columns, self.feature_index, device=device)
        self.linear_model_2 = Linear(
            linear_feature_columns, self.feature_index, device=device)

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model_1.parameters(), l2=l2_reg_linear)
        self.add_regularization_weight(self.linear_model_2.parameters(), l2=l2_reg_linear)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)


        self.out = PredictionLayer(task, )
        self.to(device)

        # parameters of callbacks
        self._is_graph_network = True  # used for ModelCheckpoint
        self.stop_training = False  # used for EarlyStopping
        self.history = History()

    def init_embedding(self, embeding_name, weight):
        ids = weight[:, 0].long().cpu().numpy().tolist()
        origin_weight = self.embedding_dict[embeding_name].weight
        for i,id in enumerate(ids):
            origin_weight[id, :] = weight[i, 1:]
        self.embedding_dict[embeding_name].weight = nn.Parameter(origin_weight)

    def compile(self, optimizer,
                snap=False,
                loss=None,
                metrics=None,
                cycle_epoch=1,
                lr=0.001
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer, lr=lr)
        self.lr_scheduler = None
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def fit(self, x=None, y=None, extra_data=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, current_epoch=1):

        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index if feature in x.keys()]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = MyDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)),
            torch.from_numpy(y),
            extra_data,
            self.feature_index
            )

        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size, num_workers=11)
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        print(f"lr: {optim.state_dict()['param_groups'][0]['lr']}")
        for epoch in range(initial_epoch+1, epochs+1):
            callbacks.on_epoch_begin(epoch)
            loss_epoch = 0
            total_loss_epoch = 0
            bar = tqdm(train_loader)
            tloss = []
            for step, (x1, y1) in enumerate(bar):
                x1 = x1.to(self.device).float()
                y1 = y1.to(self.device).float()
                y_pred = model(x1)
                optim.zero_grad()
                loss = loss_func(torch.cat(y_pred, dim=1), y1, reduction='mean')
                reg_loss = self.get_regularization_loss()
                total_loss = loss + reg_loss + self.aux_loss
                loss_epoch += loss.item()
                total_loss_epoch += total_loss.item()
                tloss.append(total_loss.item())
                total_loss.backward()
                optim.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                # if ((current_epoch - 1) * len(train_loader) + step) >= 35000 and ((current_epoch - 1) * len(train_loader) + step) % 500 == 0:
                #     print("Updates the SWA running averages of all optimized parameters.")
                #     optim.update_swa()
                bar.set_postfix(loss=np.array(tloss).mean())

        # if ((current_epoch - 1) * len(train_loader) + step) >= 35000:
        #     print("Swaps the values of the optimized variables and swa buffers.")
        #     optim.swap_swa_sgd()  # 模型权重求SWA平均
        return self.history

    def evaluate(self, x, y, batch_size=256):
        """
        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            try:
                temp = metric_fun(y, pred_ans)
            except Exception:
                temp = 0
            finally:
                eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, extra_data, batch_size=256):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index if feature in x.keys()]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = MyDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)),
            None,
            extra_data,
            self.feature_index
        )
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size, num_workers=8)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(tqdm(test_loader)):
                x1 = x_test[0].to(self.device).float()
                y_pred = model(x1, test=True)
                # y_pred_1 = torch.cat(y_pred_1, dim=1).cpu().data.numpy()
                # y_pred_2 = torch.cat(y_pred_2, dim=1).cpu().data.numpy()
                # # y_pred = (np.split(y_pred, 2, axis=1)[0]+np.split(y_pred, 2, axis=1)[1])/2.0
                # # y_pred = np.split(y_pred, 2, axis=1)[1]
                # y_pred = 0.5*y_pred_1 + 0.5*y_pred_2
                pred_ans.append(torch.cat(y_pred, dim=1).cpu().data.numpy())

        return np.concatenate(pred_ans).astype("float64")

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _get_optim(self, optimizer, useSWA=False, useLookahead=False, lr=0.1):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=lr)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters(), lr=lr)
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters(), lr=lr)
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters(), lr=lr)
            elif optimizer == 'adamw':
                optim = torch.optim.AdamW(self.parameters(), lr=lr)
            elif optimizer == 'lamb':
                import torch_optimizer as optimizer
                optim = optimizer.Lamb(self.parameters(), lr=lr)
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                self.metrics_names.append(metric)
        return metrics_

    def compute_input_dim(self, feature_columns, exclude_feature=None, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: 0 if exclude_feature!=None and exclude_feature in x.name else x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, exclude_feature=None, suport_sparse=True, support_dense=True):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        varlen_sparse_embedding_list, sparse_embedding_list = [], []

        if suport_sparse and len(varlen_sparse_feature_columns+sparse_feature_columns)>0:
            sparse_embedding_list = [embedding_dict[feat.embedding_name](
                X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
                feat in sparse_feature_columns]

            varlen_sparse_embedding_list = get_varlen_pooling_list(self.embedding_dict, X, self.feature_index,
                                                                   varlen_sparse_feature_columns, self.device)

        if support_dense:
            dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                                dense_feature_columns if exclude_feature is None or exclude_feature not in feat.name]
        else:
            dense_value_list = []

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

def get_varlen_pooling_list(embedding_dict, features, feature_index, varlen_sparse_feature_columns, device):
    varlen_sparse_embedding_list = []

    for feat in varlen_sparse_feature_columns:
        seq_emb = embedding_dict[feat.embedding_name](
            features[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long())
        if feat.length_name is None:
            seq_mask = features[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long() != 0

            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=True, device=device)(
                [seq_emb, seq_mask])
        else:
            seq_mask = features[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long() != 106444

            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=True, device=device)(
                [seq_emb, seq_mask])
        varlen_sparse_embedding_list.append(emb)
    return varlen_sparse_embedding_list