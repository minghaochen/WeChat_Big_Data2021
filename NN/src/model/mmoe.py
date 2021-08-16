import sys
import os

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../third_party'))
from deepctr_torch.inputs import *
from deepctr_torch.layers import *
from deepctr_torch.layers.core import *
from mtl_basemodel import *
from deepctr_torch.layers.sequence import AttentionSequencePoolingLayer


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


class MMoE_V1(MyBaseModel):

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, num_experts, tasks,
                 expert_dim, use_mmoe=True, num_tasks=1, task_dnn_units=None,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0., init_std=0.0001, seed=1024,
                 dnn_dropout=0.,
                 dnn_activation='relu', dnn_use_bn=False, device='cpu', gpus=None):

        super(MMoE_V1, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
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
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns, exclude_feature='static'), dnn_hidden_units,
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
        self.to(device)

    def forward(self, X, test=True):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict,
                                                                                  exclude_feature='static')
        liner_logit = self.linear_model(X)

        task_outputs = []
        if self.use_dnn and self.use_mmoe:
            dnn_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            mmoe_outs = self.mmoe(dnn_output)
            if self.task_dnn_units is not None:
                mmoe_outs = [self.task_dnn[i](mmoe_out) for i, mmoe_out in enumerate(mmoe_outs)]
            for mmoe_out, dense, out in zip(mmoe_outs, self.denses, self.outs):
                logit = dense(mmoe_out) + liner_logit
                output = out(logit, test)
                task_outputs.append(output)
        return task_outputs


class MMoE_V2(MyBaseModel):

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, num_experts, tasks,
                 expert_dim, use_mmoe=True, num_tasks=1, task_dnn_units=None, dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0., init_std=0.0001, seed=1024,
                 dnn_dropout=0., use_inner=True, use_outter=True, kernel_type='mat',
                 dnn_activation='relu', dnn_use_bn=False, device='cpu', gpus=None):

        super(MMoE_V2, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
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
        self.use_inner = use_inner
        self.use_outter = use_outter
        self.kernel_type = kernel_type
        product_out_dim = 0
        num_inputs = self.compute_input_dim(dnn_feature_columns, exclude_feature='static', include_dense=False,
                                            feature_group=True)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        if self.use_inner:
            product_out_dim += num_pairs
            self.innerproduct = InnerProductLayer(device=device)

        if self.use_outter:
            product_out_dim += num_pairs
            self.outterproduct = OutterProductLayer(
                num_inputs, self.embedding_size, kernel_type=kernel_type, device=device)

        if self.use_dnn:
            self.dnn = DNN(product_out_dim + self.compute_input_dim(dnn_feature_columns, exclude_feature='static'),
                           dnn_hidden_units,
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
        self.to(device)

    def forward(self, X, test=True):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict,
                                                                                  exclude_feature='static')
        liner_logit = self.linear_model(X)

        linear_signal = torch.flatten(
            concat_fun(sparse_embedding_list), start_dim=1)

        if self.use_inner:
            inner_product = torch.flatten(
                self.innerproduct(sparse_embedding_list), start_dim=1)

        if self.use_outter:
            outer_product = self.outterproduct(sparse_embedding_list)

        if self.use_outter and self.use_inner:
            product_layer = torch.cat(
                [linear_signal, inner_product, outer_product], dim=1)
        elif self.use_outter:
            product_layer = torch.cat([linear_signal, outer_product], dim=1)
        elif self.use_inner:
            product_layer = torch.cat([linear_signal, inner_product], dim=1)
        else:
            product_layer = linear_signal

        task_outputs = []
        if self.use_dnn and self.use_mmoe:
            dnn_input = combined_dnn_input([product_layer], dense_value_list)
            dnn_output = self.dnn(dnn_input)
            mmoe_outs = self.mmoe(dnn_output)
            if self.task_dnn_units is not None:
                mmoe_outs = [self.task_dnn[i](mmoe_out) for i, mmoe_out in enumerate(mmoe_outs)]
            for mmoe_out, dense, out in zip(mmoe_outs, self.denses, self.outs):
                logit = dense(mmoe_out) + liner_logit
                output = out(logit, test)
                task_outputs.append(output)
        return task_outputs


class MMoE_V3(MyBaseModel):

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, num_experts, tasks,
                 expert_dim, use_mmoe=True, num_tasks=1, task_dnn_units=None,
                 dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0., init_std=0.0001, seed=1024,
                 dnn_dropout=0.,
                 dnn_activation='relu', dnn_use_bn=False, device='cpu', gpus=None):

        super(MMoE_V3, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
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
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns, exclude_feature='static') + self.embedding_size,
                           dnn_hidden_units,
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

        bi_dropout = 0.
        self.bi_pooling = BiInteractionPooling()
        self.bi_dropout = bi_dropout
        if self.bi_dropout > 0:
            self.dropout = nn.Dropout(bi_dropout)
        self.to(device)

    def forward(self, X, test=True):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict,
                                                                                  exclude_feature='static')
        liner_logit = self.linear_model(X)
        fm_input = torch.cat(sparse_embedding_list, dim=1)
        bi_out = self.bi_pooling(fm_input)
        if self.bi_dropout:
            bi_out = self.dropout(bi_out)
        bi_out = torch.cat(sparse_embedding_list + [bi_out], dim=1)
        task_outputs = []
        if self.use_dnn and self.use_mmoe:
            dnn_input = combined_dnn_input([bi_out], dense_value_list)
            dnn_output = self.dnn(dnn_input)
            mmoe_outs = self.mmoe(dnn_output)
            if self.task_dnn_units is not None:
                mmoe_outs = [self.task_dnn[i](mmoe_out) for i, mmoe_out in enumerate(mmoe_outs)]
            for mmoe_out, dense, out in zip(mmoe_outs, self.denses, self.outs):
                logit = dense(mmoe_out) + liner_logit
                output = out(logit, test)
                task_outputs.append(output)
        return task_outputs

class MMoE_V4(MyBaseModel):

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, num_experts, tasks,
                 expert_dim, use_mmoe=True, num_tasks=1, task_dnn_units=None, dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0., init_std=0.0001, seed=1024,
                 dnn_dropout=0., use_inner=True, use_outter=True, kernel_type='mat',
                 dnn_activation='relu', dnn_use_bn=False, device='cpu', gpus=None):

        super(MMoE_V4, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
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
        self.use_inner = use_inner
        self.use_outter = use_outter
        self.kernel_type = kernel_type
        product_out_dim = 0
        num_inputs = self.compute_input_dim(dnn_feature_columns, exclude_feature='static', include_dense=False,
                                            feature_group=True)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        if self.use_inner:
            product_out_dim += num_pairs
            self.innerproduct = InnerProductLayer(device=device)

        if self.use_outter:
            product_out_dim += num_pairs
            self.outterproduct = OutterProductLayer(
                num_inputs, self.embedding_size, kernel_type=kernel_type, device=device)

        if self.use_dnn:
            self.dnn_1 = DNN(product_out_dim + self.compute_input_dim(dnn_feature_columns, exclude_feature='static'),
                           dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_2 = DNN(self.compute_input_dim(dnn_feature_columns, exclude_feature='static'),
                           dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)
            # torch.nn.init.normal_(self.dnn_linear.weight)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn_1.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn_2.named_parameters()), l2=l2_reg_dnn)

            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        if self.task_dnn_units is not None:
            self.task_dnn = nn.ModuleList([DNN(expert_dim, task_dnn_units + (expert_dim,)) for i in range(num_tasks)])

        self.mmoe1 = MMOELayer(num_tasks=num_tasks, num_experts=num_experts, input_dim=dnn_hidden_units[-1],
                              output_dim=expert_dim, device=device).to(device)
        self.mmoe2 = MMOELayer(num_tasks=num_tasks, num_experts=num_experts, input_dim=dnn_hidden_units[-1],
                              output_dim=expert_dim, device=device).to(device)
        self.tower_network1 = nn.ModuleList(
            [nn.Linear(expert_dim, 1, bias=False).to(device) for i, task in enumerate(self.tasks)])
        for i in range(len(self.tower_network1)):
            torch.nn.init.normal_(self.tower_network1[i].weight)

        self.tower_network2 = nn.ModuleList(
            [nn.Linear(expert_dim, 1, bias=False).to(device) for i, task in enumerate(self.tasks)])
        for i in range(len(self.tower_network2)):
            torch.nn.init.normal_(self.tower_network2[i].weight)
        self.outs = [PredictionLayer(task, ).to(device) for task in self.tasks]
        self.weight = torch.nn.Parameter(torch.tensor([0.5]*num_tasks), requires_grad=True)
        self.to(device)

    def forward(self, X, test=True):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict,
                                                                                  exclude_feature='static')
        liner_logit_1 = self.linear_model_1(X)
        liner_logit_2 = self.linear_model_2(X)

        linear_signal = torch.flatten(
            concat_fun(sparse_embedding_list), start_dim=1)

        if self.use_inner:
            inner_product = torch.flatten(
                self.innerproduct(sparse_embedding_list), start_dim=1)

        if self.use_outter:
            outer_product = self.outterproduct(sparse_embedding_list)

        if self.use_outter and self.use_inner:
            product_layer = torch.cat(
                [linear_signal, inner_product, outer_product], dim=1)
        elif self.use_outter:
            product_layer = torch.cat([linear_signal, outer_product], dim=1)
        elif self.use_inner:
            product_layer = torch.cat([linear_signal, inner_product], dim=1)
        else:
            product_layer = linear_signal

        task_outputs_1, task_outputs_2 = [], []
        dnn_input_1 = combined_dnn_input([product_layer], dense_value_list)
        dnn_input_2 = combined_dnn_input(sparse_embedding_list, dense_value_list)
        dnn_output_1 = self.dnn_1(dnn_input_1)
        dnn_output_2 = self.dnn_2(dnn_input_2)
        mmoe_outs_1 = self.mmoe1(dnn_output_1)
        mmoe_outs_2 = self.mmoe2(dnn_output_2)
        for mmoe_out_1, tower_1, out in zip(mmoe_outs_1, self.tower_network1, self.outs):
            logit = tower_1(mmoe_out_1) + liner_logit_1
            output = out(logit, test)
            task_outputs_1.append(output)
        for mmoe_out_2, tower_2, out in zip(mmoe_outs_2, self.tower_network2, self.outs):
            logit = tower_2(mmoe_out_2) + liner_logit_2
            output = out(logit, test)
            task_outputs_2.append(output)

        task_outputs = self.weight * torch.cat(task_outputs_1, dim=1) + (1-self.weight) * torch.cat(task_outputs_2, dim=1)
        # print(self.weight.data)
        return task_outputs

class MMoE_V5(MyBaseModel):

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, num_experts, tasks,
                 expert_dim, use_mmoe=True, num_tasks=1, task_dnn_units=None, dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0., init_std=0.0001, seed=1024,
                 dnn_dropout=0., use_inner=True, use_outter=True, kernel_type='mat',
                 dnn_activation='relu', dnn_use_bn=False, device='cpu', gpus=None):

        super(MMoE_V5, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
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
        self.use_inner = use_inner
        self.use_outter = use_outter
        self.kernel_type = kernel_type
        product_out_dim = 0
        num_inputs = self.compute_input_dim(dnn_feature_columns, exclude_feature='static', include_dense=False,
                                            feature_group=True)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        if self.use_inner:
            product_out_dim += num_pairs
            self.innerproduct = InnerProductLayer(device=device)

        if self.use_outter:
            product_out_dim += num_pairs
            self.outterproduct = OutterProductLayer(
                num_inputs, self.embedding_size, kernel_type=kernel_type, device=device)

        if self.use_dnn:
            self.dnn_1 = DNN(product_out_dim + self.compute_input_dim(dnn_feature_columns, exclude_feature='static'),
                           dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_2 = DNN(self.compute_input_dim(dnn_feature_columns, exclude_feature='static'),
                           dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)
            # torch.nn.init.normal_(self.dnn_linear.weight)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn_1.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn_2.named_parameters()), l2=l2_reg_dnn)

            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        if self.task_dnn_units is not None:
            self.task_dnn = nn.ModuleList([DNN(expert_dim, task_dnn_units + (expert_dim,)) for i in range(num_tasks)])

        self.mmoe1 = MMOELayer(num_tasks=num_tasks, num_experts=num_experts, input_dim=dnn_hidden_units[-1],
                              output_dim=expert_dim, device=device).to(device)
        self.mmoe2 = MMOELayer(num_tasks=num_tasks, num_experts=num_experts, input_dim=dnn_hidden_units[-1],
                              output_dim=expert_dim, device=device).to(device)
        self.tower_network1 = nn.ModuleList(
            [nn.Linear(expert_dim, 1, bias=False).to(device) for i, task in enumerate(self.tasks)])
        for i in range(len(self.tower_network1)):
            torch.nn.init.normal_(self.tower_network1[i].weight)

        self.tower_network2 = nn.ModuleList(
            [nn.Linear(expert_dim, 1, bias=False).to(device) for i, task in enumerate(self.tasks)])
        for i in range(len(self.tower_network2)):
            torch.nn.init.normal_(self.tower_network2[i].weight)
        self.outs = [PredictionLayer(task, ).to(device) for task in self.tasks]
        self.weight = torch.nn.Parameter(torch.tensor([0.5] * num_tasks), requires_grad=True)
        self.to(device)

    def forward(self, X1, X2, test=True):
        sparse_embedding_list_1, dense_value_list_1 = self.input_from_feature_columns(X1, self.dnn_feature_columns,
                                                                                  self.embedding_dict,
                                                                                  exclude_feature='static')
        sparse_embedding_list_2, dense_value_list_2 = self.input_from_feature_columns(X2, self.dnn_feature_columns,
                                                                                  self.embedding_dict,
                                                                                  exclude_feature='static')
        liner_logit_1 = self.linear_model_1(X1)
        liner_logit_2 = self.linear_model_2(X2)

        linear_signal = torch.flatten(
            concat_fun(sparse_embedding_list_1), start_dim=1)

        if self.use_inner:
            inner_product = torch.flatten(
                self.innerproduct(sparse_embedding_list_1), start_dim=1)

        if self.use_outter:
            outer_product = self.outterproduct(sparse_embedding_list_1)

        if self.use_outter and self.use_inner:
            product_layer = torch.cat(
                [linear_signal, inner_product, outer_product], dim=1)
        elif self.use_outter:
            product_layer = torch.cat([linear_signal, outer_product], dim=1)
        elif self.use_inner:
            product_layer = torch.cat([linear_signal, inner_product], dim=1)
        else:
            product_layer = linear_signal

        task_outputs_1, task_outputs_2 = [], []
        dnn_input_1 = combined_dnn_input([product_layer], dense_value_list_1)
        dnn_input_2 = combined_dnn_input(sparse_embedding_list_2, dense_value_list_2)
        dnn_output_1 = self.dnn_1(dnn_input_1)
        dnn_output_2 = self.dnn_2(dnn_input_2)
        mmoe_outs_1 = self.mmoe1(dnn_output_1)
        mmoe_outs_2 = self.mmoe2(dnn_output_2)
        for mmoe_out_1, tower_1, out in zip(mmoe_outs_1, self.tower_network1, self.outs):
            logit = tower_1(mmoe_out_1) + liner_logit_1
            output = out(logit, test)
            task_outputs_1.append(output)
        for mmoe_out_2, tower_2, out in zip(mmoe_outs_2, self.tower_network2, self.outs):
            logit = tower_2(mmoe_out_2) + liner_logit_2
            output = out(logit, test)
            task_outputs_2.append(output)

        task_outputs = self.weight * torch.cat(task_outputs_1, dim=1) + (1 - self.weight) * torch.cat(task_outputs_2, dim=1)
        return task_outputs, task_outputs_1, task_outputs_2

