
from collections import Counter
import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F

# word_embedding = nn.Embedding(2, 10) 
# rnn = nn.LSTM(10, 20, 2, batch_first=True)

# x = [torch.tensor([1]*i) for i in range(8, 3, -1)]
# # pad
# x = pad_sequence(x, batch_first=True)
# # x: tensor([[1, 1, 1, 1, 1, 1, 1, 1],
# #         [1, 1, 1, 1, 1, 1, 1, 0],
# #         [1, 1, 1, 1, 1, 1, 0, 0],
# #         [1, 1, 1, 1, 1, 0, 0, 0],
# #         [1, 1, 1, 1, 0, 0, 0, 0]])

# # get length of seq
# lens = torch.sum(x.ne(0), dim=1)
# # lens: tensor([8, 7, 6, 5, 4])

# # index to embedding
# embed = word_embedding(x)
# # embed: torch.Size([5, 8, 10])

# # # input directly
# # output, _ = rnn(embed)
# # # output: torch.Size([5, 8, 20])

# # pack
# x = pack_padded_sequence(embed, lens, batch_first=True)
# output, (hn, cn) = rnn(x)
# # output: PackSequence

# out_pad, lens_pad = pad_packed_sequence(output, batch_first=True)
# print(out_pad.size())

## 交叉熵实验
# loss = nn.CrossEntropyLoss()
# x = torch.randn(3, 5, 5, requires_grad=True)

# target = torch.empty(3, 5, dtype=torch.long).random_(5)
# print(target)
# output = loss(x, target)
# print(output)

# # log_softmax
# x = torch.randn(1, 3, 3, requires_grad=True)
# print(x)
# x1 = x[0,:,0]
# print(x1)
# print(F.softmax(x1, dim=0))
# x2 = F.softmax(x, dim=1)
# print(x2)

# # mask
# words = [torch.tensor([1]*i) for i in range(8, 5, -1)]
# # pad
# words = pad_sequence(words, batch_first=True)
# # print(words.tolist())
# print(words)
# mask = words.ne(0)
# print(mask)
# x = torch.empty(3, 8, 8).random_(5)
# print(x)
# x.masked_fill_(~mask.unsqueeze(1), float('-inf'))
# print(x)
# mask[:, 0] = 0
# print(mask)
# print(x[mask])
# heads = torch.empty(3, 8).random_(8)
# print(heads)
# print(heads[mask])

# # max-pooling
# x = torch.empty(6, 20).random_(20)
# print(x)
# y, _ = torch.max(x, dim=0)
# print(y, y.size())

# input1 = torch.randn(10, 20).unsqueeze(1)
# input2 = torch.randn(3, 20).unsqueeze(0)

# output1 = F.cosine_similarity(input1, input2, dim=2)
# # output2 = F.cosine_similarity(input1, input3, dim=1)
# # print('output2: ',output2)
# similarity = 1 - (output1 + 1) / 2
# print(similarity)

# values, indices = torch.min(similarity, dim=1)
# # print(values, indices)

# mask = indices.eq(torch.arange(3).unsqueeze(-1))
# print(mask)

# input1 = input1.squeeze(1)
# for i in range(3):
#     print(input1[mask[i]].mean(0))
# # print(input1)
# # input1 = input1.unsqueeze(0)

# # print(input1.masked_select(mask).view(-1, 20))
# # # rs = input1[mask]
# # # print(rs)
# # # print(input1.masked_select(mask.unsqueeze(-1)))
# l = [5,6,7,8]
# a = torch.arange(4)
# for i in a:
#     print(l[i])



class Sentence(object):

    def __init__(self):
        pass

class Copus():

    def __init__(self, sentences):
        
        self.sentences = sentences

sentences = [Sentence() for i in range(10)]
train_U = Copus(sentences)
print(train_U.sentences)
center_indices = torch.arange(1, 8, 2)
print(center_indices)

sub_s = [train_U.sentences[i] for i in center_indices]
for s in sub_s:
    train_U.sentences.remove(s)

for i in train_U.sentences:
    if i in []