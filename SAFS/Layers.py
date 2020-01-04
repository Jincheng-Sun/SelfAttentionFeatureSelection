import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from SAFS.ShuffleAlgorithms import cross_shuffle


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
                query {Tensor, shape: [batch, d_k, d_out]} -- query
                key {Tensor, shape: [batch, d_k, n_candidate]} -- key
                value {Tensor, shape: [batch, d_v, n_candidate]} -- value

            Returns:
                output {Tensor, shape [n_head * batch, n_depth, n_vchannel * d_features] -- output
                attn {Tensor, shape [n_head * batch, n_depth, n_depth] -- reaction attention
        '''
        attn = torch.bmm(query.transpose(2, 1), key)  # [batch, d_out, n_candidate]
        # How should we set the temperature
        attn = attn / self.temperature

        attn = self.softmax(attn)  # softmax over d_f1
        attn = self.dropout(attn)
        output = torch.bmm(attn, value.transpose(2, 1))  # [batch, d_out, d_v]

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
    def __init__(self, d_features, d_out, kernel, stride, d_k, d_v, n_replica, dropout=0.1):
        super().__init__()
        self.d_features = d_features
        self.d_out = d_out
        self.stride = np.ceil(np.divide(d_features, d_out)).astype(int)
        self.n_replica = n_replica
        self.shuffled_index = cross_shuffle(d_features, n_replica)

        self.query = nn.Conv1d(1, d_k, kernel, self.stride, bias=False, padding=1)
        self.key = nn.Conv1d(1, d_k, kernel, stride, bias=False, padding=1)
        self.value = nn.Conv1d(1, d_v, kernel, stride, bias=False, padding=1)
        self.conv = nn.Conv1d(d_v, 1, 1, 1, bias=False)

        nn.init.xavier_normal(self.query.weight)
        nn.init.xavier_normal(self.key.weight)
        nn.init.xavier_normal(self.value.weight)
        nn.init.xavier_normal(self.conv.weight)


        self.attention = ScaledDotProduction(temperature=1)

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

        residual = features

        query = self.layer_norm(features)

        # d_features_ceil = d_out * stride
        #
        # features = F.pad(features, (0, d_features_ceil - d_features), value=0)  # shape: [batch, d_features_ceil]

        shuffled_features = features[:, shuffled_index]  # shape: [batch, n_replica * d_features]

        query = self.query(query.unsqueeze(1))  # shape: [batch, d_k, d_out]
        key = self.key(
            shuffled_features.unsqueeze(1))  # shape: [batch, d_k, n_candidate], n_candidate = (n_replica * d_features) / stride
        value = self.value(shuffled_features.unsqueeze(1))  # shape: [batch, d_v, n_candidate]

        output, attn = self.attention(query, key, value)  # shape: [batch, d_out, d_v], [batch, d_out, n_candidate]
        output = output.transpose(2, 1).contiguous()  # shape: [batch, d_v, d_out]
        output = self.conv(output).view(-1, d_out)  # shape: [batch, d_out]
        output = self.dropout(output)

        if d_features == d_out:
            output += residual

        ### Use Bottleneck? ###
        output = self.bottleneck(output)

        return output, attn