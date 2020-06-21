import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProduction(nn.Module):
    '''Scaled Dot Production'''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value):
        '''
            Arguments:
                query {Tensor, shape: [batch, n_subsets, d_k]} -- query
                key {Tensor, shape: [batch, d_k, n_subsets]} -- key
                value {Tensor, shape: [batch, n_subsets, sub_size]} -- value

            Returns:
                output {Tensor, shape [n_head * batch, n_depth, n_vchannel * d_features] -- output
                attn {Tensor, shape [n_head * batch, n_depth, n_depth] -- reaction attention
        '''
        attn = torch.bmm(query, key)  # [batch, d_out, n_candidate]
        # How should we set the temperature
        attn = attn / self.temperature

        attn = self.softmax(attn)  # softmax over d_f1
        attn = self.dropout(attn)
        output = torch.bmm(attn, value)  # [batch, d_out, d_v]

        return output, attn


class Bottleneck(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        residual = features
        features = self.layer_norm(features)

        features = self.w_2(F.relu(self.w_1(features)))
        features = self.dropout(features)
        features += residual

        return features


class SelfAttentionLayer(nn.Module):
    def __init__(self, f_shuffle, d_features, d_out, kernel, stride, d_k, d_v, n_replica, dropout=0.1):
        super().__init__()
        self.d_features = d_features
        self.d_k = d_k
        self.d_v = d_v
        self.d_out = d_out
        self.stride_self = torch.ceil(np.divide(d_features, d_out)).astype(int)
        self.n_replica = n_replica
        self.shuffled_index = f_shuffle(d_features, n_replica)

        self.query = nn.Conv1d(1, d_k, kernel, self.stride_self, bias=False)
        self.key = nn.Conv1d(1, d_k, kernel, stride, bias=False)
        self.value = nn.Conv1d(1, d_v, kernel, stride, bias=False)
        self.conv = nn.Conv1d(d_v, 1, 1, 1, bias=False)

        nn.init.xavier_normal(self.query.weight)
        nn.init.xavier_normal(self.key.weight)
        nn.init.xavier_normal(self.value.weight)
        nn.init.xavier_normal(self.conv.weight)

        self.padding = nn.ConstantPad1d((0, self.stride_self * (d_out - 1) + kernel - d_features), 0)

        self.n_candidate = np.ceil(np.divide(d_features, stride)).astype(int)

        self.same_pad = (self.n_candidate - 1) * stride + kernel
        self.padding2d = nn.ConstantPad2d((0, self.same_pad - d_features, 0, 0), 0)

        self.attention = ScaledDotProduction(temperature=torch.sqrt(self.stride_self))

        self.layer_norm = nn.LayerNorm(d_features)
        self.dropout = nn.Dropout(dropout)

        ### Use Bottleneck? ###
        self.bottleneck = Bottleneck(d_out, d_out)

    def forward(self, features):
        '''
            Arguments:
                feature_1 {Tensor, shape [batch, d_features]} -- features

            Returns:
                output {Tensor, shape [batch, d_features]} -- output
                attn {Tensor, shape [n_head * batch, d_features, d_features]} -- self attention
        '''
        d_features, d_out, n_replica, shuffled_index \
            = self.d_features, self.d_out, self.n_replica, self.shuffled_index

        if d_features == d_out:
            residual = features

        query = self.layer_norm(features)

        # d_features_ceil = d_out * stride
        #
        # features = F.pad(features, (0, d_features_ceil - d_features), value=0)  # shape: [batch, d_features_ceil]

        shuffled_features = features[:, shuffled_index]  # shape: [batch, n_replica * d_features]
        shuffled_features = shuffled_features.view(-1, n_replica, d_features)
        shuffled_features = self.padding2d(shuffled_features).view(-1, self.same_pad)

        query = self.padding(query)

        query = self.query(query.unsqueeze(1))  # shape: [batch, d_k, d_out]
        key = self.key(
            shuffled_features.unsqueeze(1)).view(-1, n_replica, self.d_k,
                                                 self.n_candidate)  # shape: [batch, d_k, n_candidate], n_candidate = (n_replica * d_features) / stride
        value = self.value(shuffled_features.unsqueeze(1)).view(-1, n_replica, self.d_v,
                                                                self.n_candidate)  # shape: [batch, d_v, n_candidate]

        key = key.transpose(2, 1).contiguous().view(-1, self.d_k, n_replica * self.n_candidate)
        value = value.transpose(2, 1).contiguous().view(-1, self.d_v, n_replica * self.n_candidate)

        output, attn = self.attention(query.transpose(2, 1), key,
                                      value.transpose(2, 1))  # shape: [batch, d_out, d_v], [batch, d_out, n_candidate]
        output = output.transpose(2, 1).contiguous()  # shape: [batch, d_v, d_out]
        output = self.conv(output).view(-1, d_out)  # shape: [batch, d_out]
        output = self.dropout(output)

        if d_features == d_out:
            output += residual

        ### Use Bottleneck? ###
        output = self.bottleneck(output)

        return output, attn


class SelfAttentionLayer_V2(nn.Module):
    def __init__(self, f_shuffle, seeds, m, d, n, kernel, d_k, d_v, h, dropout=0.1):
        super().__init__()
        self.d_features = m
        self.d_out = d
        self.n_subsets = n
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = h
        self.shuffled_index = f_shuffle(m, h, seeds)

        k = np.ceil(np.divide(m, n)).astype(int)  # Subset size
        self.sub_size = k
        m_new = k * n

        self.query = nn.Linear(k, h * d_k, bias=False)
        self.key = nn.Conv1d(h, h * d_k, groups=h, kernel_size=k, stride=k,
                             bias=False)  # Act as a linear layer for multi heads
        self.value_1 = nn.Conv1d(h, h * d_v, groups=h, kernel_size=kernel, bias=False)
        self.value_2 = nn.Conv1d(h * d_v, h, groups=h, kernel_size=1, bias=False)
        self.d_out_per_subset = np.divide(d, n).astype(int)
        self.fc = nn.Linear(h * k, self.d_out_per_subset, bias=False)

        nn.init.xavier_normal(self.query.weight)
        nn.init.xavier_normal(self.key.weight)
        nn.init.xavier_normal(self.value_1.weight)
        nn.init.xavier_normal(self.value_2.weight)
        nn.init.xavier_normal(self.fc.weight)

        self.q_pad = nn.ConstantPad1d((0, m_new - m), 0)
        self.k_pad = nn.ConstantPad2d((0, m_new - m, 0, 0), 0)
        self.v_pad = nn.ConstantPad2d((0, kernel - 1, 0, 0), 0)

        self.attention = ScaledDotProduction(temperature=np.sqrt(m))

        self.layer_norm = nn.LayerNorm(m)
        self.dropout = nn.Dropout(dropout)

        ## Use Bottleneck? ###
        self.bottleneck = Bottleneck(k * h, k)

    def forward(self, features):
        '''
            Arguments:
                feature_1 {Tensor, shape [batch, d_features]} -- features

            Returns:
                output {Tensor, shape [batch, d_features]} -- output
                attn {Tensor, shape [n_head * batch, d_features, d_features]} -- self attention
        '''
        d_features, n_subsets, sub_size, n_heads, shuffled_index \
            = self.d_features, self.n_subsets, self.sub_size, self.n_heads, self.shuffled_index

        # if d_features == d_out:
        #     residual = features

        query = self.layer_norm(features)

        # d_features_ceil = d_out * stride
        #
        # features = F.pad(features, (0, d_features_ceil - d_features), value=0)  # shape: [batch, d_features_ceil]

        shuffled_features = features[:, shuffled_index]  # shape: [batch, n_heads * d_features]
        shuffled_features = shuffled_features.view(-1, n_heads, d_features)
        shuffled_features = self.k_pad(shuffled_features)

        query = self.q_pad(query)
        query = self.query(query.view(-1, n_subsets, sub_size))  # shape: [batch, d_k, d_out]
        query = query.view(-1, n_subsets, n_heads, self.d_k).transpose(1, 2).contiguous().view(-1, n_subsets, self.d_k)

        key = self.key(
            shuffled_features)  # shape: [batch, d_k, n_candidate], n_candidate = (n_heads * d_features) / stride
        key = key.view(-1, self.d_k, n_subsets)  # shape: [batch, d_k, d_out]

        value = shuffled_features.view(-1, n_heads, n_subsets, sub_size).transpose(1, 2).contiguous().view(-1, n_heads,
                                                                                                           sub_size)
        value_residual = value
        value = self.v_pad(value)
        value = self.value_1(value)  # shape: [batch, d_v, n_candidate]
        value = self.value_2(F.relu(value))
        value = value + value_residual
        value = value.view(-1, n_subsets, n_heads, sub_size).transpose(1, 2).contiguous().view(-1, n_subsets, sub_size)

        output, attn = self.attention(query, key, value)  # shape: [batch, d_out, d_v], [batch, d_out, n_candidate]
        output = output.view(-1, n_heads, n_subsets, sub_size).transpose(1, 2).contiguous().view(-1, n_subsets,
                                                                                                 n_heads * sub_size)  # shape: [batch, d_v, d_out]
        output = self.fc(output)  # shape: [batch, d_out]
        output = self.dropout(output).view(-1, self.n_subsets * self.d_out_per_subset)


        # ### Use Bottleneck? ###
        # output = self.bottleneck(output)

        return output, attn
