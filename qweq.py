

import torch.utils.data as Data

import torch


import numpy as np

import torch.nn as nn
def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table)

# c= get_sinusoid_encoding_table(4, 5)
# print(c)
# pos_emb = nn.Embedding.from_pretrained(c)
a= torch.randn(1,5)
print(a)
print(a.size())
a=torch.tensor([1, 2, 3, 4])
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
for i in a.data:
    j = i.item()
    b = idx2word[j]
    print(b)