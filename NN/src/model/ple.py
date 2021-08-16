import sys
import os

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../third_party'))
from deepctr_torch.inputs import *
from deepctr_torch.layers import *
from deepctr_torch.layers.sequence import AttentionSequencePoolingLayer
from mtl_basemodel_v2 import *


class PLELayer(nn.Module):

    def __init__(self, num_tasks, num_experts, input_dim, output_dim, selectors, device):
        super(PLELayer, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        self.selectors = selectors

        # share experts
        w = torch.Tensor(torch.rand(input_dim, self.num_experts * self.output_dim))
        torch.nn.init.normal_(w)
        self.share_expert_kernel = nn.Parameter(w).to(device)

        # task experts
        self.task_expert_kernel = []
        for i in range(self.num_tasks):
            w = torch.Tensor(torch.rand(input_dim, self.num_experts * self.output_dim))
            torch.nn.init.normal_(w)
            self.task_expert_kernel.append(nn.Parameter(w).to(device))

        # task gate
        self.task_gate_kernels = []
        for i in range(self.num_tasks):
            w = torch.Tensor(torch.rand(input_dim, self.selectors * self.num_experts))
            torch.nn.init.normal_(w)
            self.task_gate_kernels.append(nn.Parameter(w).to(device))

        # share gate
        w = torch.Tensor(torch.rand(input_dim, (self.num_tasks * self.num_experts + self.num_experts)))
        torch.nn.init.normal_(w)
        self.share_gate_kernel = nn.Parameter(w).to(device)

        # self.gate_activation = nn.Softmax(dim=-1)
        self.expert_activation = nn.ReLU()

    def forward(self, gate_output_shared_final, gate_output_task_final):
        share_expert_out = torch.mm(gate_output_shared_final, self.share_expert_kernel)
        share_expert_out = torch.reshape(share_expert_out, (-1, self.output_dim, self.num_experts))
        # share_expert_out = self.expert_activation(share_expert_out)

        task_expert_out = []
        for i in range(self.num_tasks):
            expert_out = torch.mm(gate_output_task_final[i], self.task_expert_kernel[i])
            expert_out = torch.reshape(expert_out, (-1, self.output_dim, self.num_experts))
            # expert_out = self.expert_activation(expert_out)
            task_expert_out.append(expert_out)

        task_gate_output = []
        for i in range(self.num_tasks):
            gate_out = torch.mm(gate_output_task_final[i], self.task_gate_kernels[i])
            gate_out = torch.nn.functional.softmax(gate_out, dim=1)
            gate_out = torch.unsqueeze(gate_out, dim=1).repeat(1, self.output_dim, 1)
            gate_expert_output = torch.cat([share_expert_out, task_expert_out[i]], dim=2)
            gate_output_task = torch.sum(torch.mul(gate_expert_output, gate_out), dim=2)  # sum 好于 mean
            task_gate_output.append(gate_output_task)

        share_gate_out = torch.mm(gate_output_shared_final, self.share_gate_kernel)
        share_gate_out = torch.nn.functional.softmax(share_gate_out, dim=1)
        share_gate_out = torch.unsqueeze(share_gate_out, dim=1).repeat(1, self.output_dim, 1)
        share_gate_expert_output = torch.cat([share_expert_out] + task_expert_out, dim=2)
        share_gate_out = torch.sum(torch.mul(share_gate_expert_output, share_gate_out), dim=2)

        return share_gate_out, task_gate_output


# class PLELayer(nn.Module):
#
#     def __init__(self, num_tasks, num_experts, input_dim, output_dim, selectors, device):
#         super(PLELayer, self).__init__()
#         self.num_experts = num_experts
#         self.num_tasks = num_tasks
#         self.output_dim = output_dim
#         self.selectors = selectors
#
#         # share experts
#         w = torch.Tensor(torch.rand(input_dim, self.num_experts * self.output_dim))
#         torch.nn.init.normal_(w)
#         self.share_expert_kernel = nn.Parameter(w).to(device)
#
#         # task experts
#         self.task_expert_kernel = []
#         for i in range(self.num_tasks):
#             w = torch.Tensor(torch.rand(input_dim, self.num_experts * self.output_dim))
#             torch.nn.init.normal_(w)
#             self.task_expert_kernel.append(nn.Parameter(w).to(device))
#
#         # task gate
#         self.task_gate_kernels = []
#         for i in range(self.num_tasks):
#             w = torch.Tensor(torch.rand(input_dim, self.selectors * self.num_experts))
#             torch.nn.init.normal_(w)
#             self.task_gate_kernels.append(nn.Parameter(w).to(device))
#
#         # share gate
#         w = torch.Tensor(torch.rand(input_dim, (self.num_tasks * self.num_experts + self.num_experts)))
#         torch.nn.init.normal_(w)
#         self.share_gate_kernel = nn.Parameter(w).to(device)
#
#         # self.gate_activation = nn.Softmax(dim=-1)
#         self.expert_activation = nn.ReLU()
#
#     def forward(self, gate_output_shared_final, gate_output_task_final):
#         share_expert_out = torch.mm(gate_output_shared_final, self.share_expert_kernel)
#         share_expert_out = share_expert_out.view(-1, self.num_experts, self.output_dim)
#         # share_expert_out = self.expert_activation(share_expert_out)
#
#         task_expert_out = []
#         for i in range(self.num_tasks):
#             expert_out = torch.mm(gate_output_task_final[i], self.task_expert_kernel[i])
#             expert_out = expert_out.view(-1, self.num_experts, self.output_dim)
#             # expert_out = self.expert_activation(expert_out)
#             task_expert_out.append(expert_out)
#
#         task_gate_output = []
#         for i in range(self.num_tasks):
#             gate_out = torch.mm(gate_output_task_final[i], self.task_gate_kernels[i])
#             gate_out = torch.nn.functional.softmax(gate_out, dim=1)
#             gate_expert_output = torch.cat([share_expert_out, task_expert_out[i]], dim=1)
#             gate_output_task = torch.einsum('be,beu ->beu', gate_out, gate_expert_output)
#             gate_output_task = gate_output_task.sum(dim=1)
#             task_gate_output.append(gate_output_task)
#
#         share_gate_out = torch.mm(gate_output_shared_final, self.share_gate_kernel)
#         share_gate_out = torch.nn.functional.softmax(share_gate_out, dim=1)
#         share_gate_expert_output = torch.cat([share_expert_out]+task_expert_out, dim=1)
#         share_gate_out = torch.einsum('be,beu ->beu', share_gate_out, share_gate_expert_output)
#         share_gate_out = share_gate_out.sum(dim=1)
#
#         return share_gate_out, task_gate_output

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


class PLE(MyBaseModel):

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, num_experts, tasks,
                 expert_dim, use_mmoe=True, num_tasks=1, task_dnn_units=None,
                 dnn_hidden_units=(256, 128), selectors=2,
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0., init_std=0.0001, seed=1024,
                 dnn_dropout=0., dnn_activation='relu', dnn_use_bn=False, device='cpu', gpus=None):

        super(PLE, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
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
        self.num_tasks = num_tasks
        self.task_dnn_units = task_dnn_units
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)

            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
            # torch.nn.init.normal_(self.dnn_linear.weight)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        if self.task_dnn_units is not None:
            self.task_dnn = nn.ModuleList([DNN(expert_dim, task_dnn_units + (expert_dim,)) for i in range(num_tasks)])

        if self.use_mmoe:
            self.ple_1 = PLELayer(num_tasks=num_tasks, num_experts=num_experts, input_dim=dnn_hidden_units[-1],
                                  output_dim=expert_dim, device=device, selectors=selectors).to(device)
            # self.ple_2 = PLELayer(num_tasks=num_tasks, num_experts=num_experts, input_dim=expert_dim,
            #                       output_dim=expert_dim, device=device, selectors=selectors).to(device)
            self.tower_network = nn.ModuleList(
                [nn.Linear(expert_dim, 1, bias=False).to(device) for i, task in enumerate(self.tasks)])
            for i in range(len(self.tower_network)):
                torch.nn.init.normal_(self.tower_network[i].weight)
        self.outs = [PredictionLayer(task, ).to(device) for task in self.tasks]
        self.to(device)

    def forward(self, X, test=True):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        liner_logit = self.linear_model(X)

        task_outputs = []
        if self.use_dnn and self.use_mmoe:
            dnn_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)

            ple_out_shared, ple_out_task_final = self.ple_1(dnn_output, [dnn_output for i in range(self.num_tasks)])
            # ple_out_shared, ple_out_task_final =  self.ple_2(ple_out_shared, ple_out_task_final)

            if self.task_dnn_units is not None:
                ple_out_task_final = [self.task_dnn[i](mmoe_out) for i, mmoe_out in enumerate(ple_out_task_final)]

            for ple_out, tower, out in zip(ple_out_task_final, self.tower_network, self.outs):
                logit = tower(ple_out) + liner_logit
                output = out(logit, test)
                task_outputs.append(output)
        return task_outputs


class PLE_V2(MyBaseModel):

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, num_experts, tasks,
                 expert_dim, use_mmoe=True, num_tasks=1, task_dnn_units=None,
                 dnn_hidden_units=(256, 128), selectors=2,
                 use_inner=True, use_outter=True, kernel_type='mat',
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0., init_std=0.0001, seed=1024,
                 dnn_dropout=0., dnn_activation='relu', dnn_use_bn=False, device='cpu', gpus=None):

        super(PLE_V2, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
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
        self.num_tasks = num_tasks
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

            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
            # torch.nn.init.normal_(self.dnn_linear.weight)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        if self.task_dnn_units is not None:
            self.task_dnn = nn.ModuleList([DNN(expert_dim, task_dnn_units + (expert_dim,)) for i in range(num_tasks)])

        if self.use_mmoe:
            self.ple_1 = PLELayer(num_tasks=num_tasks, num_experts=num_experts, input_dim=dnn_hidden_units[-1],
                                  output_dim=expert_dim, device=device, selectors=selectors).to(device)
            # self.ple_2 = PLELayer(num_tasks=num_tasks, num_experts=num_experts, input_dim=expert_dim,
            #                       output_dim=expert_dim, device=device, selectors=selectors).to(device)
            self.tower_network = nn.ModuleList(
                [nn.Linear(expert_dim, 1, bias=False).to(device) for i, task in enumerate(self.tasks)])
            for i in range(len(self.tower_network)):
                torch.nn.init.normal_(self.tower_network[i].weight)
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

            ple_out_shared, ple_out_task_final = self.ple_1(dnn_output, [dnn_output for i in range(self.num_tasks)])
            # ple_out_shared, ple_out_task_final =  self.ple_2(ple_out_shared, ple_out_task_final)

            if self.task_dnn_units is not None:
                ple_out_task_final = [self.task_dnn[i](mmoe_out) for i, mmoe_out in enumerate(ple_out_task_final)]

            for ple_out, tower, out in zip(ple_out_task_final, self.tower_network, self.outs):
                logit = tower(ple_out) + liner_logit
                output = out(logit, test)
                task_outputs.append(output)
        return task_outputs