import torch.nn as nn
from SAFS.Layers import SelfAttentionLayer, SelfAttentionLayer_V2


class SelfAttentionFeatureSelection(nn.Module):
    def __init__(self, f_shuffle, d_features, d_out_list, kernel, stride, d_k, d_v, n_replica):
        super().__init__()

        d_out_list.insert(0, d_features)

        self.layers = nn.ModuleList([
            SelfAttentionLayer(f_shuffle, d_in, d_out, kernel=kernel, stride=stride, d_k=d_k, d_v=d_v,
                               n_replica=n_replica,
                               ) for d_in, d_out in zip(d_out_list, d_out_list[1:])])

        ## TODO: Is layer_norm at the end really necessary?
        self.layer_norm = nn.LayerNorm(d_out_list[-1], eps=1e-6)

    def forward(self, features):
        self_attn_list = []

        for layer in self.layers:
            features, self_attn = layer(
                features)
            self_attn_list += [self_attn]

        features = self.layer_norm(features)

        return features, self_attn_list


class LinearClassifier(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super(LinearClassifier, self).__init__()

        self.fc1 = nn.Linear(d_in, d_hid)
        self.activation = nn.functional.leaky_relu
        self.fc2 = nn.Linear(d_hid, d_out)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class SelfAttentionFeatureSelection_V2(nn.Module):
    def __init__(self, f_shuffle, d_features, n_subset_list, d_out_list, kernel, stride, d_k, d_v, h):
        super().__init__()

        d_out_list.insert(0, d_features)

        self.layers = nn.ModuleList([
            SelfAttentionLayer_V2(f_shuffle, d_in, d_out, n_sub, kernel, d_k, d_v, h,
                                  ) for n_sub, d_in, d_out in zip(n_subset_list, d_out_list, d_out_list[1:])])

        ## TODO: Is layer_norm at the end really necessary?
        self.layer_norm = nn.LayerNorm(d_out_list[-1], eps=1e-6)

    def forward(self, features):
        self_attn_list = []

        for layer in self.layers:
            features, self_attn = layer(
                features)
            self_attn_list += [self_attn]

        features = self.layer_norm(features)

        return features, self_attn_list
