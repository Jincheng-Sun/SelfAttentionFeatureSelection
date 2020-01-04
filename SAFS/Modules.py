import torch.nn as nn
from SAFS.Layers import SelfAttentionLayer

class SelfAttentionFeatureSelection(nn.Module):
    def __init__(self, d_features, d_out_list, kernel, stride):
        super().__init__()

        d_out_list.insert(0, d_features)

        self.layers = nn.ModuleList([
            SelfAttentionLayer(d_in, d_out, kernel=kernel, stride=stride, d_k=32, d_v=32, n_replica=8,
                               ) for d_in, d_out in zip(d_out_list, d_out_list[1:])])

    def forward(self, features):
        self_attn_list = []

        for layer in self.layers:
            features, self_attn = layer(
                features)
            self_attn_list += [self_attn]

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
