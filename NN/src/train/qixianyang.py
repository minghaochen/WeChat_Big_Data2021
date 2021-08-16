# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os

BASE_DIR = 'model'
sys.path.append(os.path.join(BASE_DIR, '../model'))
from mmoe import *

sys.path.append(os.path.join(BASE_DIR, '../third_party'))
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import random

sys.path.append(os.path.join(BASE_DIR, '..'))
from evaluation import uAUC, evaluate_deepctr
import time as ti
import torch

sys.path.append(os.path.join(BASE_DIR, '../../config'))
from conf import *
import sys
import os
from deepctr_torch.inputs import *
from deepctr_torch.layers import *
from deepctr_torch.layers.core import *
from mtl_basemodel import *
from deepctr_torch.layers.sequence import AttentionSequencePoolingLayer
from torch.utils.data import Dataset

MAX_SEQ = 10


class MMOEDataset(Dataset):
    def __init__(self, x, y, x_userid, x_date_, history_sequence):
        self.x = x
        self.userid = x_userid
        self.date = x_date_
        self.history_seq = history_sequence
        self.userid_date = self.history_seq.index
        self.y = y
        self.use_cache = True
        self.last_userid = 0
        self.last_date = 0
        self.history_like_seq = np.zeros(MAX_SEQ, dtype=int)
        self.history_readcomment_seq = np.zeros(MAX_SEQ, dtype=int)
        self.history_feedid_seq = np.zeros(MAX_SEQ, dtype=int)
        self.history_authorid_seq = np.zeros(MAX_SEQ, dtype=int)

    def __len__(self):
        return self.userid.shape[0]

    def __getitem__(self, idx):

        x_userid = self.userid[idx]
        present_date_ = self.date[idx]

        if (self.use_cache) & (x_userid == self.last_userid) & (present_date_ == self.last_date):
            history_like_seq = self.last_history_like_seq
            history_readcomment_seq = self.last_history_readcomment_seq
            history_feedid_seq = self.last_history_feedid_seq
            history_authorid_seq = self.last_history_authorid_seq
        else:
            history_like_seq = np.zeros(MAX_SEQ, dtype=int)
            history_readcomment_seq = np.zeros(MAX_SEQ, dtype=int)
            history_feedid_seq = np.zeros(MAX_SEQ, dtype=int)
            history_authorid_seq = np.zeros(MAX_SEQ, dtype=int)
            for last_date_ in range(present_date_ - 1, 1, -1):
                if (x_userid, last_date_) in self.userid_date:
                    ## dataFrame() 针对双index索引做过优化
                    feedid_seq, authorid_seq, like_seq, readcomment_seq = self.history_seq[(x_userid, last_date_)]
                    seq_length = len(feedid_seq)
                    if seq_length > MAX_SEQ:
                        history_feedid_seq = feedid_seq[-MAX_SEQ:]
                        history_authorid_seq = authorid_seq[-MAX_SEQ:]
                        history_like_seq = like_seq[-MAX_SEQ:]
                        history_readcomment_seq = readcomment_seq[-MAX_SEQ:]
                    else:
                        history_feedid_seq[-seq_length:] = feedid_seq
                        history_authorid_seq[-seq_length:] = authorid_seq
                        history_like_seq[-seq_length:] = like_seq
                        history_readcomment_seq[-seq_length:] = readcomment_seq
                    break
            if self.use_cache:
                self.last_userid = x_userid
                self.last_date = present_date_
                self.last_history_like_seq = history_like_seq
                self.last_history_readcomment_seq = history_readcomment_seq
                self.last_history_feedid_seq = history_feedid_seq
                self.last_history_authorid_seq = history_authorid_seq
        dict_data = {
            'x': self.x[idx],
            'y': self.y[idx],
            'readcomment_seq': history_readcomment_seq,
            'like_seq': history_like_seq,
            'feedid_seq': history_feedid_seq,
            'authorid_seq': history_authorid_seq
        }
        return dict_data


class MMOELayer(nn.Module):

    def __init__(self, num_tasks, num_experts, input_dim, output_dim, device):
        super(MMOELayer, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        w = torch.Tensor(torch.rand(input_dim, self.num_experts * self.output_dim))
        torch.nn.init.normal_(w)
        self.expert_kernel = nn.Parameter(w).to(device)
        self.gate_kernels = []
        for i in range(self.num_tasks):
            w = torch.Tensor(torch.rand(input_dim, self.num_experts))
            torch.nn.init.normal_(w)
            self.gate_kernels.append(nn.Parameter(w).to(device))

    def forward(self, inputs):
        outputs = []
        expert_out = torch.mm(inputs, self.expert_kernel)
        expert_out = torch.reshape(expert_out, (-1, self.output_dim, self.num_experts))
        for i in range(self.num_tasks):
            gate_out = torch.mm(inputs, self.gate_kernels[i])
            gate_out = torch.nn.functional.softmax(gate_out, dim=1)
            gate_out = torch.unsqueeze(gate_out, dim=1).repeat(1, self.output_dim, 1)
            output = torch.sum(torch.mul(expert_out, gate_out), dim=2)
            outputs.append(output)
        return outputs


class DNN(nn.Module):

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class PredictionLayer(nn.Module):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")

        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, X, test=False):
        output = X
        if self.use_bias:
            output += self.bias
        if self.task == "binary":
            if test:
                output = torch.sigmoid(output)
            else:
                output = output
        return output


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

        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device)

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.out = PredictionLayer(task, )
        self.to(device)

        # parameters of callbacks
        self._is_graph_network = True  # used for ModelCheckpoint
        self.stop_training = False  # used for EarlyStopping
        self.history = History()

    def compile(self, optimizer,
                train_len,
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
        if snap:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim,
                                                                                     T_0=train_len * cycle_epoch,
                                                                                     T_mult=1, eta_min=1e-7,
                                                                                     last_epoch=-1)
        else:
            self.lr_scheduler = None
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def fit(self, x=None, y=None, extra_data=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0,
            validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, current_epoch=1):
        print('start overfitting...........')

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

        if isinstance(x, dict):
            x_date_ = x['date_'].values
            x_userid = x['userid'].values

            x = [x[feature] for feature in self.feature_index if feature != 'date_']
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        ## 改动1
        train_tensor_data = MMOEDataset(np.concatenate(x, axis=-1), y, x_userid, x_date_, history_sequence)
        #         train_tensor_data = Data.TensorDataset(
        #             torch.from_numpy(np.concatenate(x, axis=-1)),
        #             torch.from_numpy(y)
        #             )
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
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size, num_workers=14)
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
        for epoch in range(initial_epoch + 1, epochs + 1):
            callbacks.on_epoch_begin(epoch)
            loss_epoch = 0
            total_loss_epoch = 0
            bar = tqdm(train_loader)
            tloss = []
            for step, data in enumerate(bar):
                x_train = data['x']
                y_train = data['y']
                x = x_train.to(self.device).float()
                y = y_train.to(self.device).float()
                ## 改动 2
                y_pred = model(x, X_seq=data)
                optim.zero_grad()
                loss = loss_func(torch.cat(y_pred, dim=1), y, reduction='mean')
                reg_loss = self.get_regularization_loss()
                total_loss = loss + reg_loss + self.aux_loss
                loss_epoch += loss.item()
                total_loss_epoch += total_loss.item()
                tloss.append(total_loss.item())
                total_loss.backward()
                optim.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                bar.set_postfix(loss=np.array(tloss).mean())

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
            x_date_ = x['date_'].values
            x_userid = x['userid'].values
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = MMOEDataset(np.concatenate(x, axis=-1), np.zeros(x[0].shape[0]), x_userid, x_date_,
                                  history_sequence)
        #         tensor_data = Data.TensorDataset(
        #             torch.from_numpy(np.concatenate(x, axis=-1)),
        #         )
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(tqdm(test_loader)):
                x = x_test['x'].to(self.device).float()
                y_pred = model(x, X_seq=x_test, test=True)
                y_pred = torch.cat(y_pred, dim=1).cpu().data.numpy()
                pred_ans.append(y_pred)

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

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
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


class MMoE(MyBaseModel):

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, num_experts, tasks,
                 expert_dim, use_mmoe=True, num_tasks=1, task_dnn_units=None,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0., init_std=0.0001, seed=1024,
                 dnn_dropout=0.,
                 dnn_activation='relu', dnn_use_bn=False, device='cpu', gpus=None):

        super(MMoE, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                   l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, device=device,
                                   gpus=gpus)
        if num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if len(tasks) != num_tasks:
            raise ValueError("num_tasks must be equal to the length of tasks")
        for task in tasks:
            if task not in ['binary', 'regression']:
                raise ValueError("task must be binary or regression, {} is illegal".format(task))
        self.tasks = tasks
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        self.use_mmoe = use_mmoe
        self.task_dnn_units = task_dnn_units

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)
            # torch.nn.init.normal_(self.dnn_linear.weight)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        if self.task_dnn_units is not None:
            self.task_dnn = nn.ModuleList([DNN(expert_dim, task_dnn_units + (expert_dim,)) for i in range(num_tasks)])

        if self.use_mmoe:
            self.mmoe = MMOELayer(num_tasks=num_tasks, num_experts=num_experts, input_dim=dnn_hidden_units[-1],
                                  output_dim=expert_dim, device=device).to(device)
            self.denses = nn.ModuleList(
                [nn.Linear(expert_dim, 1, bias=False).to(device) for i, task in enumerate(self.tasks)])
            for i in range(len(self.denses)):
                torch.nn.init.normal_(self.denses[i].weight)
        self.outs = [PredictionLayer(task, ).to(device) for task in self.tasks]

        ## 处理序列信息相关embedding和layer
        self.embed_dim = 20

        self.like_embedding = nn.Embedding(2, self.embed_dim)
        self.readcomment_embedding = nn.Embedding(2, self.embed_dim)
        self.feedid_embedding = nn.Embedding(feedid_vocabulary_size, self.embed_dim)
        self.authorid_embedding = nn.Embedding(authorid_vocabulary_size, self.embed_dim)
        self.seq_emb_lr = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.seq_attention = nn.MultiheadAttention(self.embed_dim, num_heads=4)
        #         self.seq_attention_lr = nn.Linear(self.embed_dim, len(self.tasks))

        self.seq_attention_lr = nn.Linear(self.embed_dim, 1)
        self.logit_lr = nn.Linear(3, 1)
        self.to(device)

    ## 改动3
    def get_seq_emb(self, X_seq):
        readcomment_seq = X_seq['readcomment_seq'].long().to(device)
        like_seq = X_seq['like_seq'].long().to(device)
        feedid_seq = X_seq['feedid_seq'].long().to(device)
        authorid_seq = X_seq['authorid_seq'].long().to(device)
        like_emb = self.like_embedding(like_seq)
        readcomment_emb = self.readcomment_embedding(readcomment_seq)
        feedid_emb = self.feedid_embedding(feedid_seq)
        authorid_emb = self.feedid_embedding(authorid_seq)
        #         feedid_emb = feedid_emb + like_emb + readcomment_emb
        #         authorid_emb = authorid_emb + like_emb + readcomment_emb
        #         seq_emb = torch.cat([feedid_emb, authorid_emb], dim=-1)
        #         return seq_emb
        return feedid_emb, like_emb

    def get_seq_output(self, X_seq, feedid):
        seq_emb, like_emb = self.get_seq_emb(X_seq)
        #         seq_emb = self.seq_emb_lr(seq_emb)

        k = seq_emb.transpose(0, 1)
        v = like_emb.transpose(0, 1)
        q = feedid.transpose(0, 1)

        attn_output, _ = self.seq_attention(q, k, v)

        attn_output = attn_output.transpose(0, 1)
        seq_out = self.seq_attention_lr(attn_output).squeeze(-2)
        return seq_out

    def forward(self, X, X_seq=None, test=True):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        ## 序列输出
        if X_seq:
            seq_out = self.get_seq_output(X_seq, sparse_embedding_list[1])

        liner_logit = self.linear_model(X)
        task_outputs = []
        if self.use_dnn and self.use_mmoe:
            dnn_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            mmoe_outs = self.mmoe(dnn_output)
            if self.task_dnn_units is not None:
                mmoe_outs = [self.task_dnn[i](mmoe_out) for i, mmoe_out in enumerate(mmoe_outs)]
            for i, (mmoe_out, dense, out) in enumerate(zip(mmoe_outs, self.denses, self.outs)):
                logit = dense(mmoe_out) + liner_logit + seq_out
                output = out(logit, test)
                task_outputs.append(output)
        return task_outputs


def seed_everything(seed):  # seed
    """
    Seeds basic parameters for reproductibility of results
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 6666
seed_everything(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEBUG_MODE = False
USE_DR = True
FEED_DIM = 20
EPOCHS = 3
BATCH = 1024
cycle_epoch = 3
valid_days = [14]
seed_list = [1024]
test_preds = []

model_save_dir = os.path.join(ROOT_PATH, 'model/mmoe_checkpoints/')
submit_dir = os.path.join(ROOT_PATH, "single_model_result")
if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)

t1 = ti.time()

ROOT_PATH = '../data/'
TEST_FILE = '../data/test_data_chusai_a.csv'
submit = pd.read_csv(TEST_FILE)[['userid', 'feedid']]
target = ["read_comment", "like", "click_avatar", "forward"]
sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
dense_features = ['videoplayseconds', 'device']
USE_FEAT = sparse_features + dense_features + ['date_']
train_all = pd.read_csv(os.path.join(ROOT_PATH, "train_all_data_chusai.csv"), usecols=USE_FEAT + target)
test = pd.read_csv(os.path.join(ROOT_PATH, "test_data_chusai_b.csv"), usecols=USE_FEAT)
test['date_'] = 15

print('load data finish..........')


def reID(df_data, sparse_features):
    dict_unique_id = {}
    for feature in sparse_features:
        unique_id = df_data[feature].unique()
        dict_unique_id[feature] = dict(zip(unique_id, range(len(unique_id))))
    return dict_unique_id


all_data = train_all.append(test)
dict_reid = reID(all_data, sparse_features)
del all_data
for feature in sparse_features:
    train_all[feature] = train_all[feature].map(dict_reid[feature])
    test[feature] = test[feature].map(dict_reid[feature])
test = test[USE_FEAT]
tra = train_all
actions_of_train = tra[target].reset_index(drop=True)
tra = tra[USE_FEAT]
train_len = tra.shape[0]
all_data = pd.concat((tra, test)).reset_index(drop=True)
all_data[dense_features] = all_data[dense_features].fillna(0)
all_data['videoplayseconds'] = np.log(all_data["videoplayseconds"] + 1.0)
mms = MinMaxScaler(feature_range=(0, 1))
all_data[[i for i in dense_features if 'multi_mode_feed_' not in i]] = mms.fit_transform(
    all_data[[i for i in dense_features if 'multi_mode_feed_' not in i]])

fixlen_feature_columns = [SparseFeat(feat, all_data[feat].max() + 1, embedding_dim=FEED_DIM)
                          for feat in sparse_features] + [DenseFeat(feat, 1, ) for feat in dense_features]

feedid_vocabulary_size = fixlen_feature_columns[1][1]
authorid_vocabulary_size = fixlen_feature_columns[2][1]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(dnn_feature_columns)

tra, test = all_data.iloc[:train_len], all_data.iloc[train_len:]
tra = pd.concat((tra, actions_of_train), axis=1)

# 4.Define Model,train,predict and evaluate
device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

t2 = ti.time()
ts = (t2 - t1) / 3600.0
print(f"prepare data take time:{ts:.3f}")

history_sequence = train_all.query('num_target>0').groupby(['userid', 'date_']).apply(lambda x: (x['feedid'].values,
                                                                                                 x['authorid'].values,
                                                                                                 x['like'].values,
                                                                                                 x[
                                                                                                     'read_comment'].values))
for i, d in enumerate(valid_days):
    print(f"valid day {d}")
    DEBUG_MODE = True
    if d == 14 and DEBUG_MODE == False:
        train, train_index = tra, tra.index.tolist()
    else:
        train, train_index = tra[~(tra['date_'] == d)], tra[~(tra['date_'] == d)].index.tolist()
    valid, valid_index = tra[tra['date_'] == d], tra[tra['date_'] == d].index.tolist()
    test_index = test.index.tolist()

    userid_list = valid['userid'].astype(str).tolist()
    train_model_input = {name: train[name] for name in feature_names + ['date_']}
    valid_model_input = {name: valid[name] for name in feature_names + ['date_']}
    test_model_input = {name: test[name] for name in feature_names + ['date_']}

    train_labels = [train[y].values for y in target]
    train_labels = np.concatenate([np.expand_dims(i, axis=1) for i in train_labels], axis=1)
    val_labels = [valid[y].values for y in target]
    val_labels = np.concatenate([np.expand_dims(i, axis=1) for i in val_labels], axis=1)
    test_labels = np.zeros(test_model_input['userid'].shape[0])
    weight_auc = 0.

    for seed in seed_list:
        model = MMoE(
            linear_feature_columns=linear_feature_columns,
            dnn_feature_columns=dnn_feature_columns,
            dnn_hidden_units=(512, 256),
            task_dnn_units=None,
            num_experts=8, num_tasks=4, expert_dim=16,
            tasks=['binary', 'binary', 'binary', 'binary'],
            device=device,
            seed=seed
        )
        train_len = round(train.shape[0] / BATCH)
        model.compile(optimizer='adagrad', train_len=train_len, snap=False, loss="binary_crossentropy",
                      metrics=["binary_crossentropy", "auc"], lr=1e-2)

        for epoch in range(1, EPOCHS + 1):
            print(f"{epoch}/{EPOCHS}")
            history = model.fit(train_model_input, train_labels, extra_data=None, batch_size=BATCH, shuffle=True
                                , epochs=1, verbose=1, current_epoch=epoch)
            val_pred_ans = model.predict(valid_model_input, extra_data=None, batch_size=BATCH * 2)
            val_uauc = evaluate_deepctr(val_labels, val_pred_ans, userid_list, target)
            weight_auc = val_uauc / len(seed_list)
            if epoch % cycle_epoch == 0:
                print(f"cycle: {epoch / cycle_epoch}/{EPOCHS / cycle_epoch}")
                model_save_path = os.path.join(model_save_dir, f'mmoe_model_fold{i + 1}.pth')
                print(model_save_path)
                torch.save(model.state_dict(), model_save_path)
                # save dnn weight
                t1 = ti.time()
                pred_ans = model.predict(test_model_input, extra_data=None, batch_size=BATCH * 5)
                t2 = ti.time()
                print(f'{len(target)}个目标行为%d条样本预测耗时（毫秒）：%.3f' % (len(test), (t2 - t1) * 1000.0))
                ts = (t2 - t1) * 1000.0 / len(test) * 2000.0
                print(f'{len(target)}个目标行为2000条样本平均预测耗时（毫秒）：%.3f' % ts)
                test_preds.append(pred_ans)
        torch.cuda.empty_cache()

# 5.生成提交文件
for i, action in enumerate(target):
    submit[action] = 0.
    for f in test_preds:
        submit[action] += (f[:, i].squeeze()) / (len(seed_list) * len(valid_days) * (EPOCHS / cycle_epoch))
submit[['userid', 'feedid'] + target].to_csv(os.path.join(submit_dir, 'mmoe_5fold.csv'), index=None,
                                             float_format='%.6f')