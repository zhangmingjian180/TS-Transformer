# -*- encoding: utf-8 -*-
'''
@File    :   layers.py
@Time    :   2022/10/24 14:30:00
@Author  :   Jian Zhang
@Contact :   zhangmingjian180@qq.com
HUST, Heilongjiang, China

@Reference : Fei gao, feig@mail.bnu.edu.cn, https://github.com/FeiGSSS/DySAT_pytorch/blob/main/models/layers.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class AddPositionalEncoding(nn.Module):
    def __init__(self, time_steps, input_dim):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.Tensor(time_steps, input_dim))
        self.xavier_init()

    def forward(self, inputs):
        x = inputs + self.position_embeddings # [N, T, F]
        return x

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)


class EncoderBlock(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 n_heads, 
                 drop=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads

        # define weights
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.xavier_init()
        
        # define function
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.ff_line1 = nn.Linear(output_dim, 4*output_dim, bias=True)  # feed forward
        self.ff_line2 = nn.Linear(4*output_dim, output_dim, bias=True)
        self.dropout1 = nn.Dropout(drop)  # dropout
        self.dropout2 = nn.Dropout(drop)
        
        if input_dim != output_dim:
            self.residual_line = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, inputs):
        # 1: Query, Key based multi-head self attention.
        q = inputs @ self.Q_embedding_weights  # [N, T, F]
        k = inputs @ self.K_embedding_weights  # [N, T, F]
        v = inputs @ self.V_embedding_weights  # [N, T, F]

        # 2: Split, concat.
        split_size = q.shape[-1] // self.n_heads
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=-1), dim=0)  # [N*h, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=-1), dim=0)  # [N*h, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=-1), dim=0)  # [N*h, T, F/h]
        
        # 3: Compute attention weights.
        a = q_ @ k_.transpose(-2, -1)  # [N*h, T, T]
        a = a / (q_.shape[-1] ** 0.5)
        a = F.softmax(a, dim=-1)
        
        # 4: Compute outputs.
        o = a @ v_  # [N*h, T, F/h]
        split_size = o.shape[0] // self.n_heads
        o = torch.cat(torch.split(o, split_size_or_sections=split_size, dim=0), dim=-1)  # [N, T, F]
       
        # 5: Add and lay normalization.
        o = self.dropout1(o)
        if self.input_dim != self.output_dim:
            inputs = self.residual_line(inputs)
        out = o + inputs
        out = self.layer_norm1(out)

        # 6: Feedforward and residual
        outputs = F.relu(self.ff_line1(out))
        outputs = self.ff_line2(outputs)
        
        # 7: Add and lay normalization.
        outputs = self.dropout2(outputs)
        outputs = outputs + out
        outputs = self.layer_norm2(outputs)
        
        return outputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)

class DecoderBlock(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 n_heads, 
                 drop=0.5):
        super().__init__()
        self.n_heads = n_heads

        # define weights
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.xavier_init()
        
        # define function
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.ff_line1 = nn.Linear(output_dim, 4*output_dim, bias=True)  # feed forward
        self.ff_line2 = nn.Linear(4*output_dim, output_dim, bias=True)
        self.dropout1 = nn.Dropout(drop)  # dropout
        self.dropout2 = nn.Dropout(drop)

    def forward(self, inputs, encoder_x):
        # 1: Query, Key based multi-head self attention.
        q = inputs @ self.Q_embedding_weights  # [N, 1, F]
        k = encoder_x @ self.K_embedding_weights  # [N, T, F]
        v = encoder_x @ self.V_embedding_weights  # [N, T, F]

        # 2: Split, concat.
        split_size = q.shape[-1] // self.n_heads
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=-1), dim=0)  # [N*h, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=-1), dim=0)  # [N*h, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=-1), dim=0)  # [N*h, T, F/h]
        
        # 3: Compute attention weights.
        a = q_ @ k_.transpose(-2, -1)  # [N*h, T, T]
        a = a / (q_.shape[-1] ** 0.5)
        a = F.softmax(a, dim=-1)
        
        # 4: Compute outputs.
        o = a @ v_  # [N*h, T, F/h]
        split_size = o.shape[0] // self.n_heads
        o = torch.cat(torch.split(o, split_size_or_sections=split_size, dim=0), dim=-1)  # [N, T, F]
       
        # 5: Add and lay normalization.
        o = self.dropout1(o)
        out = o + inputs
        out = self.layer_norm1(out)

        # 6: Feedforward and residual
        outputs = F.relu(self.ff_line1(out))
        outputs = self.ff_line2(outputs)
        
        # 7: Add and lay normalization.
        outputs = self.dropout2(outputs)
        outputs = outputs + out
        outputs = self.layer_norm2(outputs)
        
        return outputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)

class GlobalPool(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.A_embedding_weights = nn.Parameter(torch.Tensor(input_dim, 1))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.xavier_init()

    def forward(self, inputs):
        v = inputs @ self.V_embedding_weights  # [N, T, F]
        a = inputs @ self.A_embedding_weights  # [N, T, 1]

        a = a / (inputs.shape[-1] ** 0.5)
        a = a.transpose(-2, -1)
        a = F.softmax(a, dim=-1)
        
        o = a @ v

        return o

    def xavier_init(self):
        nn.init.xavier_uniform_(self.A_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)

class FCN(nn.Module):
    def __init__(self, input_dim, n_classes, linear_drop=0.5):
        super().__init__()
        self.classifier = nn.Sequential(nn.Dropout(linear_drop),
                                        nn.Linear(input_dim, n_classes))

    def forward(self, x):
        return self.classifier(x)









